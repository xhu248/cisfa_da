# CISFA

## meaning of different model
cut_atten_coseg_sum_model.py: final method;

cut_coseg_sum_model.py: w/o weight on the pcl (ncl);

cut_coseg_model.py: for gcl, update the encoder first and then the whole segmenter;

cut_model: just cut, w/o any contrastive losses;

cut_shape_coseg_model: change the order of generative and classifier model;