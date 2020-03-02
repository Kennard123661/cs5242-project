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

# same as create lmdb feature extraction list
# we do feature extraction for 10 splits

python3 utils/create_vmz_db.py \
--list_file=breakfast/test.csv \
--output_file=breakfast/test_lmbd \
--use_list=1 --use_video_id=1 --use_start_frame=1
