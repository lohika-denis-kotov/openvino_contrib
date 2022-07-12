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

#include "ngraph/op/convolution.hpp"

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
                      std::is_same_v<TBaseConvolution, ov::op::v1::GroupConvolutionBackpropData>,
                  "TBaseConvolution should be either ConvolutionBackpropData or GroupConvolutionBackpropData");

    auto convolution = std::dynamic_pointer_cast<TBaseConvolution>(m.get_match_root());
    if (!convolution) {
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
    std::shared_ptr<TBaseConvolution> new_convolution;
    if (convolution->inputs().size() == 3) {
        const ov::Output<ov::Node> &output_shape = convolution->input(2).get_source_output();
        new_convolution = std::make_shared<TBaseConvolution>(padding->output(0),
                                                             filters,
                                                             output_shape,
                                                             convolution->get_strides(),
                                                             zero_pads,
                                                             zero_pads,
                                                             convolution->get_dilations(),
                                                             ov::op::PadType::EXPLICIT);
    } else {
        new_convolution = std::make_shared<TBaseConvolution>(padding->output(0),
                                                             filters,
                                                             convolution->get_strides(),
                                                             zero_pads,
                                                             zero_pads,
                                                             convolution->get_dilations(),
                                                             ov::op::PadType::EXPLICIT);
    }

    new_convolution->set_friendly_name(convolution->get_friendly_name());
    ov::copy_runtime_info(convolution, new_convolution);
    ov::replace_node(convolution, new_convolution);

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

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvBackpropDataAsymPaddingTransformation,
                       "ConvBackpropDataAsymPaddingTransformation",
                       0);
ConvBackpropDataAsymPaddingTransformation::ConvBackpropDataAsymPaddingTransformation() {
    const auto conv = pattern::wrap_type<ov::op::v1::ConvolutionBackpropData>();

    matcher_pass_callback callback = [](pattern::Matcher &m) {
        return convolution_backprop_data_with_padding<ov::op::v1::ConvolutionBackpropData>(m);
    };
    const auto m = std::make_shared<pattern::Matcher>(conv, "ConvBackpropDataAsymPaddingTransformation");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::GroupConvBackpropDataAsymPaddingTransformation,
                       "GroupConvBackpropDataAsymPaddingTransformation",
                       0);
GroupConvBackpropDataAsymPaddingTransformation::GroupConvBackpropDataAsymPaddingTransformation() {
    const auto conv = pattern::wrap_type<ov::op::v1::GroupConvolutionBackpropData>();

    matcher_pass_callback callback = [](pattern::Matcher &m) {
        return convolution_backprop_data_with_padding<ov::op::v1::GroupConvolutionBackpropData>(m);
    };
    const auto m = std::make_shared<pattern::Matcher>(conv, "GroupConvBackpropDataAsymPaddingTransformation");
    register_matcher(m, callback);
}

}  // namespace ngraph::pass
