is_output_img = True
is_output_sift_matching = False
is_output_chosen_points = False
is_output_op = False
is_output_frame_match = True
match_standard_thresh = 40
match_front_car = 900
text_pos = 100
text_size = 1
text_color = (255, 0, 0)
mask_dir = "./mask/"
mask_car_color = (0, 103, 200)
mask_human_color = (150, 5, 60)
mask_building_color = (180, 120, 120)
mask_plate_color = (255, 5, 154)
mask_margin = 5  # pixel
mask_scale_factor = 10  # how mask will scale before and after re-sampling
mask_op_trend_thresh = 0.5
max_norm_dis = 100
max_grid = 10
root_dir = "E:/lane_modeling/the_173/"
o_f_name = "outputs"  # output folder name
# optimize
u0_cache = 1920
v0_cache = 1088
# choose bev points
down_points_num = 6
minimal_bev_dis = 40
