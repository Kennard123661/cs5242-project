# Copyright 2018-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

python tools/dense_prediction_fusion.py \
--input_dir1=/data/users/trandu/datasets/kinetics_features/rgb_ft_45450620 \
--input_dir2=/data/users/trandu/datasets/kinetics_features/of_ft_50030948 \
--alpha=0.6
