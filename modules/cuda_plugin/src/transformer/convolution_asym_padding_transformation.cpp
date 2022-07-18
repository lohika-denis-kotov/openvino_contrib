// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_asym_padding_transformation.hpp"

#include <gsl/gsl_assert>
#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/pad.hpp>
#include <transformer/nodes/fused_convolution_backprop_data.hpp>

namespace {

ov::CoordinateDiff add_two_zero_pads(const ov::CoordinateDiff &pad) {
    ov::CoordinateDiff result = pad;
    result.insert(result.begin(), 0);
    result.insert(result.begin(), 0);

    return result;
}

template <typename TBaseConvolution>
bool convolution_with_padding(ngraph::pattern::Matcher &m) {
    static_assert(std::is_same_v<TBaseConvolution, ov::op::v1::Convolution> ||
                      std::is_same_v<TBaseConvolution, ov::op::v1::GroupConvolution>,
                  "TBaseConvolution should be either Convolution or GroupConvolution");

    auto convolution = std::dynamic_pointer_cast<TBaseConvolution>(m.get_match_root());
    if (!convolution || convolution->inputs().size() != 2) {
        return false;
    }

    const auto pads_begin = add_two_zero_pads(convolution->get_pads_begin());
    const auto pads_end = add_two_zero_pads(convolution->get_pads_end());

    if (pads_begin == pads_end) {
        return false;
    }
    Expects(pads_begin.size() == pads_end.size());

    const ov::Output<ov::Node> &data = convolution->input(0).get_source_output();
    const ov::Output<ov::Node> &filters = convolution->input(1).get_source_output();

    const auto pads_begin_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{pads_begin.size()}, pads_begin.data());
    const auto pads_end_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{pads_end.size()}, pads_end.data());
    const auto padding =
        std::make_shared<ov::op::v1::Pad>(data,
                                          pads_begin_node,
                                          pads_end_node,
                                          ov::op::v0::Constant::create(data.get_element_type(), ov::Shape{}, {0}),
                                          ov::op::PadMode::CONSTANT);

    const ov::CoordinateDiff zero_pads(convolution->get_pads_begin().size(), 0);
    auto new_convolution = std::make_shared<TBaseConvolution>(padding->output(0),
                                                              filters,
                                                              convolution->get_strides(),
                                                              zero_pads,
                                                              zero_pads,
                                                              convolution->get_dilations(),
                                                              ov::op::PadType::EXPLICIT);

    new_convolution->set_friendly_name(convolution->get_friendly_name());
    ov::copy_runtime_info(convolution, new_convolution);
    ov::replace_node(convolution, new_convolution);

    return true;
}

template <typename TBaseConvolution>
bool convolution_backprop_data_with_padding(ngraph::pattern::Matcher &m) {
    static_assert(std::is_same_v<TBaseConvolution, ov::op::v1::ConvolutionBackpropData> ||
                      std::is_same_v<TBaseConvolution, ov::op::v1::GroupConvolutionBackpropData> ||
                      std::is_same_v<TBaseConvolution, CUDAPlugin::nodes::FusedConvBackpropData>,
                  "TBaseConvolution should be either ConvolutionBackpropData or GroupConvolutionBackpropData");

    auto convolution = std::dynamic_pointer_cast<TBaseConvolution>(m.get_match_root());
    if (!convolution) {
        return false;
    }

    const auto &pads_begin = convolution->get_pads_begin();
    const auto &pads_end = convolution->get_pads_end();
    if (pads_begin == pads_end) {
        return false;
    }
    Expects(pads_begin.size() == pads_end.size());

    const auto &output_padding = convolution->get_output_padding();
    const auto &strides = convolution->get_strides();
    const auto &dilations = convolution->get_dilations();

    const auto &data = convolution->input(0).get_source_output();
    const auto &input_shape = data.get_node()->output(0).get_shape();
    const auto &filters = convolution->input(1).get_source_output();

    ov::PartialShape conv_output_shape_data{};
    if constexpr (std::is_same_v<TBaseConvolution, ov::op::v1::GroupConvolutionBackpropData>) {
        conv_output_shape_data = convolution->get_convolution_output_shape();
    } else {
        conv_output_shape_data = convolution->get_output_shape();
    }
    constexpr auto num_non_spatial_dims = 2;
    ov::Shape static_conv_output_shape_data{};
    if (conv_output_shape_data.is_dynamic()) {
        conv_output_shape_data = convolution->output(0).get_shape();
        if (conv_output_shape_data.is_dynamic()) {
            return false;
        }
        auto node_conv_output_shape = conv_output_shape_data.to_shape();
        static_conv_output_shape_data =
            ov::Shape{node_conv_output_shape.begin() + num_non_spatial_dims, node_conv_output_shape.end()};
    } else {
        static_conv_output_shape_data = conv_output_shape_data.to_shape();
    }

    for (int i = 0; i < pads_begin.size(); ++i) {
        static_conv_output_shape_data[i] = pads_begin[i] + static_conv_output_shape_data[i] + pads_end[i];
    }
    auto output_shape_node = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{static_conv_output_shape_data.size()}, static_conv_output_shape_data);

    const ov::CoordinateDiff zero_pads(pads_begin.size(), 0);
    std::shared_ptr<TBaseConvolution> new_convolution;
    if constexpr (std::is_same_v<TBaseConvolution, CUDAPlugin::nodes::FusedConvBackpropData>) {
        if (convolution->inputs().size() == 4) {
            const ov::Output<ov::Node> &add = convolution->input(3).get_source_output();
            new_convolution = std::make_shared<TBaseConvolution>(data,
                                                                 filters,
                                                                 output_shape_node,
                                                                 add,
                                                                 strides,
                                                                 zero_pads,
                                                                 zero_pads,
                                                                 dilations,
                                                                 ov::op::PadType::EXPLICIT,
                                                                 output_padding);
        } else {
            const ov::Output<ov::Node> &add = convolution->input(2).get_source_output();
            new_convolution = std::make_shared<TBaseConvolution>(data,
                                                                 filters,
                                                                 output_shape_node,
                                                                 add,
                                                                 strides,
                                                                 zero_pads,
                                                                 zero_pads,
                                                                 dilations,
                                                                 ov::op::PadType::EXPLICIT,
                                                                 output_padding);
        }
    } else {
        new_convolution = std::make_shared<TBaseConvolution>(data,
                                                             filters,
                                                             output_shape_node,
                                                             strides,
                                                             zero_pads,
                                                             zero_pads,
                                                             dilations,
                                                             ov::op::PadType::EXPLICIT,
                                                             output_padding);
    }

    new_convolution->validate_and_infer_types();

    [[maybe_unused]] const auto &old_conv_shape = convolution->output(0).get_shape();
    [[maybe_unused]] const auto &new_conv_shape = new_convolution->output(0).get_shape();
    Expects(old_conv_shape != new_conv_shape);

    std::vector<int64_t> begins(num_non_spatial_dims, 0);
    for (int i = 0; i < pads_begin.size(); ++i) {
        begins.push_back(static_cast<int64_t>(pads_begin[i]));
    }
    std::vector<int64_t> ends(num_non_spatial_dims, 0);
    for (int i = 0; i < pads_end.size(); ++i) {
        ends.push_back(new_conv_shape[num_non_spatial_dims + i] - static_cast<int64_t>(pads_end[i]));
    }

    const auto slice_begin_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{begins.size()}, begins.data());
    const auto slice_end_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{ends.size()}, ends.data());

    std::vector<int64_t> strided_slice_strides(input_shape.size(), 1);
    const auto slice_strides_node = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{strided_slice_strides.size()}, strided_slice_strides.data());

    std::vector<int64_t> begin_mask(num_non_spatial_dims, 1);
    for (int i = 0; i < pads_begin.size(); ++i) {
        begin_mask.push_back(0);
    }
    std::vector<int64_t> end_mask(num_non_spatial_dims, 1);
    for (int i = 0; i < pads_end.size(); ++i) {
        end_mask.push_back(0);
    }

    const auto slice = std::make_shared<ov::op::v1::StridedSlice>(
        new_convolution, slice_begin_node, slice_end_node, slice_strides_node, begin_mask, end_mask);

    slice->set_friendly_name(convolution->get_friendly_name());
    ov::copy_runtime_info(convolution, slice);
    ov::replace_node(convolution, slice);

    [[maybe_unused]] const auto &strided_slice_shape = slice->output(0).get_shape();
    Expects(old_conv_shape == strided_slice_shape);

    return true;
}
}  // namespace

namespace ngraph::pass {

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvolutionAsymPaddingTransformation, "ConvolutionAsymPaddingTransformation", 0);

ConvolutionAsymPaddingTransformation::ConvolutionAsymPaddingTransformation() {
    const auto conv = pattern::wrap_type<ov::op::v1::Convolution>();

    matcher_pass_callback callback = [](pattern::Matcher &m) {
        return convolution_with_padding<ov::op::v1::Convolution>(m);
    };

    const auto m = std::make_shared<pattern::Matcher>(conv, "ConvolutionAsymPaddingTransformation");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::GroupConvolutionAsymPaddingTransformation,
                       "GroupConvolutionAsymPaddingTransformation",
                       0);
GroupConvolutionAsymPaddingTransformation::GroupConvolutionAsymPaddingTransformation() {
    const auto conv = pattern::wrap_type<ov::op::v1::GroupConvolution>();

    matcher_pass_callback callback = [](pattern::Matcher &m) {
        return convolution_with_padding<ov::op::v1::GroupConvolution>(m);
    };
    const auto m = std::make_shared<pattern::Matcher>(conv, "GroupConvolutionAsymPaddingTransformation");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvolutionBackpropDataAsymPaddingTransformation,
                       "ConvolutionBackpropDataAsymPaddingTransformation",
                       0);
ConvolutionBackpropDataAsymPaddingTransformation::ConvolutionBackpropDataAsymPaddingTransformation() {
    const auto conv = pattern::wrap_type<ov::op::v1::ConvolutionBackpropData>();

    matcher_pass_callback callback = [](pattern::Matcher &m) {
        return convolution_backprop_data_with_padding<ov::op::v1::ConvolutionBackpropData>(m);
    };
    const auto m = std::make_shared<pattern::Matcher>(conv, "ConvolutionBackpropDataAsymPaddingTransformation");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::GroupConvolutionBackpropDataAsymPaddingTransformation,
                       "GroupConvolutionBackpropDataAsymPaddingTransformation",
                       0);
GroupConvolutionBackpropDataAsymPaddingTransformation::GroupConvolutionBackpropDataAsymPaddingTransformation() {
    const auto conv = pattern::wrap_type<ov::op::v1::GroupConvolutionBackpropData>();

    matcher_pass_callback callback = [](pattern::Matcher &m) {
        return convolution_backprop_data_with_padding<ov::op::v1::GroupConvolutionBackpropData>(m);
    };
    const auto m = std::make_shared<pattern::Matcher>(conv, "GroupConvolutionBackpropDataAsymPaddingTransformation");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::FusedConvBackpropDataAsymPaddingTransformation,
                       "FusedConvBackpropDataAsymPaddingTransformation",
                       0);
FusedConvBackpropDataAsymPaddingTransformation::FusedConvBackpropDataAsymPaddingTransformation() {
    const auto conv = pattern::wrap_type<CUDAPlugin::nodes::FusedConvBackpropData>();

    matcher_pass_callback callback = [](pattern::Matcher &m) {
        return convolution_backprop_data_with_padding<CUDAPlugin::nodes::FusedConvBackpropData>(m);
    };
    const auto m = std::make_shared<pattern::Matcher>(conv, "FusedConvBackpropDataAsymPaddingTransformation");
    register_matcher(m, callback);
}

}  // namespace ngraph::pass
