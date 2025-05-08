"""
DATE: 15/07/2023
LAST CHANGE: 27/07/2023
AUTHOR: CHENG ZHANG


** Uncomment before using **

Only for debug using, generally the code in file wouldn't be used.
This file includes collections of debug code to keep the original files clean for reading.
The correct usage and location of each collection of debug code is specified.
"""

################################################################################################################
# Print all the embedding size                                                                                 #
# Copy and paste after line 254 in /model/model.py                                                             #
################################################################################################################
# # debug code for embedding layer
# print(lsp_embedded.size())
# print(pitch_embedded.size())
# print(code_embedded.size())
# print(gain_embedded.size())
# print(og_embed.size())


################################################################################################################
# Print all the output size in layer 1                                                                         #
# Copy and paste after line #TODO xxx in /model/model.py                                                       #
################################################################################################################
# # debug code for layer 1
# print('layer_1_full_size_out')
# print(layer_1_full_size_out.size())
# print('layer_1_down_1_out')
# print(layer_1_down_1_out.size())
# print('layer_1_down_2_out')
# print(layer_1_down_2_out.size())
# print('layer_1_down_3_out')
# print(layer_1_down_3_out.size())
# print('layer_1_down_4_out')
# print(layer_1_down_4_out.size())
# print('layer_1_down_5_out')
# print(layer_1_down_5_out.size())
# print('layer_1_up_1_out')
# print(layer_1_up_1_out.size())
# print('layer_1_up_2_out')
# print(layer_1_up_2_out.size())
# print('layer_1_up_3_out')
# print(layer_1_up_3_out.size())
# print('layer_1_up_4_out')
# print(layer_1_up_4_out.size())
# print('layer_1_up_5_out')
# print(layer_1_up_5_out.size())

# # print the final feature output sizes of layer1
# print('layer_1_feature_1')
# print(layer_1_feature_1.size())
# print('layer_1_feature_2')
# print(layer_1_feature_2.size())
# print('layer_1_feature_3')
# print(layer_1_feature_3.size())
# print('layer_1_feature_4')
# print(layer_1_feature_4.size())
# print('layer_1_feature_5')
# print(layer_1_feature_5.size())

# exit(0)


################################################################################################################
# Branch 5 cross-correlation feature element attention layer debug codes                                       #
# Copy and paste after line #TODO xxx in /model/model.py                                                       #
################################################################################################################
# # print total feature number
# print('\nlayer 1 calculated number')
# print(num_features)
# print('cross_correlation_features')
# print(cross_correlation_features.size())
# print(branch_5_attention_out.size())

# exit(0)


################################################################################################################
# Print all the output size in layer 2                                                                         #
# Copy and paste after line #TODO xxx in /model/model.py                                                       #
################################################################################################################

# # debug code for layer 2
# print('layer_2_full_size_out')
# print(layer_2_full_size_out.size())
# print('layer_2_down_1_out')
# print(layer_2_down_1_out.size())
# print('layer_2_down_2_out')
# print(layer_2_down_2_out.size())
# print('layer_2_down_3_out')
# print(layer_2_down_3_out.size())
# print('layer_2_up_1_out')
# print(layer_2_up_1_out.size())
# print('layer_2_up_2_out')
# print(layer_2_up_2_out.size())
# print('layer_2_up_3_out')
# print(layer_2_up_3_out.size())

# # print the final feature output sizes of layer 2
# print('layer_2_feature_1')
# print(layer_2_feature_1.size())
# print('layer_2_feature_2')
# print(layer_2_feature_2.size())
# print('layer_2_feature_3')
# print(layer_2_feature_3.size())

# # print total feature number
# num_features_2 = calculate_num_features(self.input_frame, self.embed_dim, 2)
# print('\nlayer 2 calculated number')
# print(num_features_2)
# print('layer_2_mix_feature')
# print(layer_2_mix_feature.size())

# exit(0)

################################################################################################################
# Print all the output size in layer 3                                                                         #
# Copy and paste after line #TODO xxx in /model/model.py                                                       #
################################################################################################################

# # debug code for layer 3
# print('layer_3_full_size_out')
# print(layer_3_full_size_out.size())
# print('layer_3_down_1_out')
# print(layer_3_down_1_out.size())
# print('layer_3_down_2_out')
# print(layer_3_down_2_out.size())
# print('layer_3_down_3_out')
# print(layer_3_down_3_out.size())
# print('layer_3_down_4_out')
# print(layer_3_down_4_out.size())
# print('layer_3_up_1_out')
# print(layer_3_up_1_out.size())
# print('layer_3_up_2_out')
# print(layer_3_up_2_out.size())
# print('layer_3_up_3_out')
# print(layer_3_up_3_out.size())
# print('layer_3_up_4_out')
# print(layer_3_up_4_out.size())
#
# # print the final feature output sizes of layer 3
# print('layer_3_feature_1')
# print(layer_3_feature_1.size())
# print('layer_3_feature_2')
# print(layer_3_feature_2.size())
# print('layer_3_feature_3')
# print(layer_3_feature_3.size())
# print('layer_3_feature_4')
# print(layer_3_feature_4.size())

# # print total feature number
# num_features_3 = calculate_num_features(self.input_frame, self.embed_dim, 3)
# print('\nlayer 3 calculated number')
# print(num_features_3)
# print('layer_3_mix_feature')
# print(layer_3_mix_feature.size())

# exit(0)

################################################################################################################
# Print all the output size in layer 4                                                                         #
# Copy and paste after line #TODO xxx in /model/model.py                                                       #
################################################################################################################

# # debug code for layer 4
# print('layer_4_full_size_out')
# print(layer_4_full_size_out.size())
# print('layer_4_down_1_out')
# print(layer_4_down_1_out.size())
# print('layer_4_down_2_out')
# print(layer_4_down_2_out.size())
# print('layer_4_down_3_out')
# print(layer_4_down_3_out.size())
# print('layer_4_down_4_out')
# print(layer_4_down_4_out.size())
# print('layer_4_up_1_out')
# print(layer_4_up_1_out.size())
# print('layer_4_up_2_out')
# print(layer_4_up_2_out.size())
# print('layer_4_up_3_out')
# print(layer_4_up_3_out.size())
# print('layer_4_up_4_out')
# print(layer_4_up_4_out.size())
#
# # print the final feature output sizes of layer 4
# print('layer_4_feature_1')
# print(layer_4_feature_1.size())
# print('layer_4_feature_2')
# print(layer_4_feature_2.size())
# print('layer_4_feature_3')
# print(layer_4_feature_3.size())
# print('layer_4_feature_4')
# print(layer_4_feature_4.size())

# # print total feature number
# num_features_4 = calculate_num_features(self.input_frame, self.embed_dim, 4)
# print('\nlayer 4 calculated number')
# print(num_features_4)
# print('layer_4_mix_feature')
# print(layer_4_mix_feature.size())

# exit(0)

################################################################################################################
# Print all the output size in layer 5                                                                         #
# Copy and paste after line #TODO xxx in /model/model.py                                                       #
################################################################################################################

# # debug code for layer 5
# print('layer_5_full_size_out')
# print(layer_5_full_size_out.size())
# print('layer_5_down_1_out')
# print(layer_5_down_1_out.size())
# print('layer_5_down_2_out')
# print(layer_5_down_2_out.size())
# print('layer_5_down_3_out')
# print(layer_5_down_3_out.size())
# print('layer_5_down_4_out')
# print(layer_5_down_4_out.size())
# print('layer_5_up_1_out')
# print(layer_5_up_1_out.size())
# print('layer_5_up_2_out')
# print(layer_5_up_2_out.size())
# print('layer_5_up_3_out')
# print(layer_5_up_3_out.size())
# print('layer_5_up_4_out')
# print(layer_5_up_4_out.size())
#
# # print the final feature output sizes of layer 5
# print('layer_5_feature_1')
# print(layer_5_feature_1.size())
# print('layer_5_feature_2')
# print(layer_5_feature_2.size())
# print('layer_5_feature_3')
# print(layer_5_feature_3.size())
# print('layer_5_feature_4')
# print(layer_5_feature_4.size())

# # print total feature number
# num_features_5 = calculate_num_features(self.input_frame, self.embed_dim, 5)
# print('\nlayer 5 calculated number')
# print(num_features_5)
# print('layer_5_mix_feature')
# print(layer_5_mix_feature.size())

# exit(0)

################################################################################################################
# Branch 6 intra-correlation feature element attention layer debug codes                                       #
# Copy and paste after line #TODO xxx in /model/model.py                                                       #
################################################################################################################

# # print total feature number
# num_features_6 = num_features_2 + num_features_3 + num_features_4 + num_features_5
# print('\nlayer 5 calculated number')
# print(num_features_6)
# print('intra_correlation_features')
# print(intra_correlation_features.size())
# print(branch_6_attention_out.size())

# exit(0)