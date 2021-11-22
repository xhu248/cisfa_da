# CISFA
This is the code for paper #5387

## run the code
```
bash cut_abd_job.sh
```

## meaning of different model
cut_atten_coseg_sum_model.py: final method;

cut_coseg_sum_model.py: w/o weight on the pcl (ncl);

cut_coseg_sequential_model.py: for gcl, update the encoder first and then the whole segmenter;

