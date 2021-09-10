// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_convolution_backprop_data.hpp"

#include <gsl/gsl_assert>
#include <ngraph/validation_util.hpp>

namespace CUDAPlugin::nodes {

FusedConvBackpropData::FusedConvBackpropData(const ngraph::Output<Node>& data_batch,
                                             const ngraph::Output<Node>& filters,
                                             const ngraph::Output<Node>& add,
                                             const ngraph::Strides& strides,
                                             const ngraph::CoordinateDiff& pads_begin,
                                             const ngraph::CoordinateDiff& pads_end,
                                             const ngraph::Strides& dilations,
                                             const ngraph::op::PadType& auto_pad,
                                             const ngraph::CoordinateDiff& output_padding)
    : ngraph::op::Op(ngraph::OutputVector{data_batch, filters, add}),
      strides_(strides),
      pads_begin_(pads_begin),
      pads_end_(pads_end),
      dilations_(dilations),
      auto_pad_(auto_pad),
      output_padding_(output_padding),
      add_shape_{add.get_shape()},
      add_type_(add.get_element_type()) {
    constructor_validate_and_infer_types();
}

FusedConvBackpropData::FusedConvBackpropData(const ngraph::Output<Node>& data_batch,
                                             const ngraph::Output<Node>& filters,
                                             const ngraph::Output<Node>& outputShape,
                                             const ngraph::Output<Node>& add,
                                             const ngraph::Strides& strides,
                                             const ngraph::CoordinateDiff& pads_begin,
                                             const ngraph::CoordinateDiff& pads_end,
                                             const ngraph::Strides& dilations,
                                             const ngraph::op::PadType& auto_pad,
                                             const ngraph::CoordinateDiff& output_padding)
    : ngraph::op::Op(ngraph::OutputVector{data_batch, filters, outputShape, add}),
      strides_(strides),
      pads_begin_(pads_begin),
      pads_end_(pads_end),
      dilations_(dilations),
      auto_pad_(auto_pad),
      output_padding_(output_padding),
      add_shape_{add.get_shape()},
      add_type_(add.get_element_type()) {
    constructor_validate_and_infer_types();
}

bool FusedConvBackpropData::visit_attributes(ngraph::AttributeVisitor& visitor) {
    visitor.on_attribute("strides", strides_);
    visitor.on_attribute("dilations", dilations_);
    visitor.on_attribute("pads_begin", pads_begin_);
    visitor.on_attribute("pads_end", pads_end_);
    visitor.on_attribute("auto_pad", auto_pad_);
    visitor.on_attribute("output_padding", output_padding_);
    return true;
}

std::shared_ptr<ngraph::Node> FusedConvBackpropData::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 3) {
        return std::make_shared<FusedConvBackpropData>(new_args.at(0),
                                                       new_args.at(1),
                                                       new_args.at(2),
                                                       strides_,
                                                       pads_begin_,
                                                       pads_end_,
                                                       dilations_,
                                                       auto_pad_,
                                                       output_padding_);
    } else {
        return std::make_shared<FusedConvBackpropData>(new_args.at(0),
                                                       new_args.at(1),
                                                       new_args.at(2),
                                                       new_args.at(3),
                                                       strides_,
                                                       pads_begin_,
                                                       pads_end_,
                                                       dilations_,
                                                       auto_pad_,
                                                       output_padding_);
    }
}

void FusedConvBackpropData::conv_validate_and_infer_types() {
    auto data_pshape = get_input_partial_shape(0);
    ngraph::element::Type delta_et = get_input_element_type(0);
    const ngraph::PartialShape& filters_pshape = get_input_partial_shape(1);
    ngraph::element::Type filters_et = get_input_element_type(1);

    bool is_output_shape_present = inputs().size() == 3;
    ngraph::PartialShape output_pshape = get_output_shape();

    ngraph::element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          ngraph::element::Type::merge(result_et, delta_et, filters_et),
                          "Element types for data batch and filters do not match (data batch element type: ",
                          delta_et,
                          ", filters element type: ",
                          filters_et,
                          ").");

    if (data_pshape.rank().is_static() && filters_pshape.rank().is_static()) {
        if (pads_begin_.size() == 0) {
            pads_begin_ = conv_default_padding(this, data_pshape, filters_pshape);
        }
        if (pads_end_.size() == 0) {
            pads_end_ = conv_default_padding(this, data_pshape, filters_pshape);
        }
        if (output_padding_.size() == 0) {
            output_padding_ = conv_default_padding(this, data_pshape, filters_pshape);
        }
        if (strides_.size() == 0) {
            strides_ = conv_default_strides(this, data_pshape, filters_pshape);
        }
        if (dilations_.size() == 0) {
            dilations_ = conv_default_strides(this, data_pshape, filters_pshape);
        }

        const auto num_spatial_dims = data_pshape.rank().get_length() - 2;

        NODE_VALIDATION_CHECK(
            this, strides_.size() == num_spatial_dims, "Strides should be defined for all and only spatial features.");

        NODE_VALIDATION_CHECK(this,
                              dilations_.size() == num_spatial_dims,
                              "Dilations should be defined for all and only spatial features.");

        NODE_VALIDATION_CHECK(this,
                              output_padding_.size() == num_spatial_dims,
                              "Output padding should be defined for all and only "
                              "spatial features.");
    }

    ngraph::PartialShape result_shape;
    if (is_output_shape_present) {
        if (output_pshape.is_static() && filters_pshape.is_static() && data_pshape.is_static()) {
            ngraph::Shape output_shape = output_pshape.to_shape();
            const ngraph::Shape data_shape = data_pshape.to_shape();
            const ngraph::Shape filters_shape = filters_pshape.to_shape();
            const size_t num_spatial_dims = data_shape.size() - 2;

            NODE_VALIDATION_CHECK(this,
                                  output_shape.size() == num_spatial_dims,
                                  "Output shape should be specified only and for "
                                  "all spatial dimensions.");

            // If auto_pad has one of following mode we infer paddings. Otherwise in
            // EXPLICIT auto_pad mode we use what is provided.
            if (auto_pad_ == ngraph::op::PadType::SAME_UPPER || auto_pad_ == ngraph::op::PadType::SAME_LOWER) {
                ngraph::opset1::infer_conv_backprop_auto_padding(
                    ngraph::Shape{std::next(data_shape.begin(), 2), std::end(data_shape)},
                    ngraph::Shape{std::next(filters_shape.begin(), 2), std::end(filters_shape)},
                    output_shape,
                    strides_,
                    dilations_,
                    auto_pad_,
                    output_padding_,
                    pads_begin_,
                    pads_end_);
            }

            // C_OUTPUT
            output_shape.insert(output_shape.begin(), filters_shape.at(1));
            // N
            output_shape.insert(output_shape.begin(), data_shape.at(0));
            output_pshape = output_shape;
        }
        set_input_is_relevant_to_shape(2);
    }
    // Deduce output shape from input spatial shape, strides, dilations, output padding
    // and padding values.
    else {
        if (auto_pad_ == ngraph::op::PadType::SAME_UPPER || auto_pad_ == ngraph::op::PadType::SAME_LOWER ||
            auto_pad_ == ngraph::op::PadType::VALID) {
            pads_begin_.assign(pads_begin_.size(), 0);
            pads_end_.assign(pads_end_.size(), 0);
        }

        if (data_pshape.rank().is_static() && filters_pshape.is_static()) {
            std::vector<ngraph::Dimension> data_shape{data_pshape}, filters_shape{filters_pshape}, output_shape;

            infer_conv_backprop_output_spatial_shape(
                std::vector<ngraph::Dimension>{std::next(data_shape.begin(), 2), std::end(data_shape)},
                std::vector<ngraph::Dimension>{std::next(filters_shape.begin(), 2), std::end(filters_shape)},
                strides_,
                dilations_,
                pads_begin_,
                pads_end_,
                output_padding_,
                output_shape);

            // C_OUTPUT
            output_shape.insert(output_shape.begin(), filters_shape.at(1));
            // N
            output_shape.insert(output_shape.begin(), data_shape.at(0));
            output_pshape = ngraph::PartialShape{output_shape};
        } else {
            output_pshape = ngraph::PartialShape::dynamic(data_pshape.rank());
        }
    }

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_output_type(0, result_et, output_pshape);
}

ngraph::PartialShape FusedConvBackpropData::get_output_shape() const {
    auto data_pshape = get_input_partial_shape(0);

    ngraph::PartialShape shape;
    if (data_pshape.rank().is_static()) {
        shape = ngraph::PartialShape{std::vector<ngraph::Dimension>(data_pshape.rank().get_length() - 2)};
    } else {
        shape = ngraph::PartialShape{std::vector<ngraph::Dimension>(strides_.size())};
    }
    bool is_output_shape_present = inputs().size() == 3;
    if (is_output_shape_present) {
        if (auto const_op = get_constant_from_source(input_value(2))) {
            shape = const_op->get_shape_val();
        } else {
            shape = ngraph::PartialShape::dynamic();
        }
    }
    return shape;
}

void FusedConvBackpropData::infer_conv_backprop_output_spatial_shape(
    const std::vector<ngraph::Dimension>& input_data_shape,
    const std::vector<ngraph::Dimension>& filters_shape,
    const ngraph::Strides& strides,
    const ngraph::Strides& dilations,
    const ngraph::CoordinateDiff& pads_begin,
    const ngraph::CoordinateDiff& pads_end,
    const ngraph::CoordinateDiff& output_padding,
    std::vector<ngraph::Dimension>& output_spatial_shape) {
    size_t num_spatial_dims = input_data_shape.size();
    NODE_VALIDATION_CHECK(this,
                          filters_shape.size() == num_spatial_dims && strides.size() == num_spatial_dims &&
                              dilations.size() == num_spatial_dims && pads_begin.size() == num_spatial_dims &&
                              pads_end.size() == num_spatial_dims && output_padding.size() == num_spatial_dims);

    for (size_t i = 0; i < num_spatial_dims; ++i) {
        if (input_data_shape[i].is_static() && filters_shape[i].is_static()) {
            int64_t val = strides[i] * (input_data_shape[i].get_length() - 1) +
                          dilations[i] * (filters_shape[i].get_length() - 1) + 1 - pads_begin[i] - pads_end[i] +
                          output_padding[i];
            output_spatial_shape.push_back(val);
        } else {
            output_spatial_shape.push_back(ngraph::Dimension::dynamic());
        }
    }
}

void FusedConvBackpropData::validate_and_infer_types() {
    conv_validate_and_infer_types();
    const auto& conv_out_shape = get_output_shape();
    const auto& element_type = get_output_element_type(0);
    //  Expects(conv_out_shape == add_shape_);
    Expects(element_type == add_type_);
}

}  // namespace CUDAPlugin::nodes
