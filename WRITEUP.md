## Questions & Answers

### Q1) How long did your first working version take on the full 50 files, and how long did your final version take? Which single change saved the most time?

I ran the first working version on 2 parquet files of about 66 videos with baseline strategy of each-frame per video. It took about 5,493 seconds on Colab L4 GPU and approx. 10k seconds on my pc while doing experiments. When extrapolated to full 50 files, the first version would have taken approx 46 hours on L4 GPU. My final version took 2.23 hours for 50 files. Adaptive frames selection saved the most time, it took 5,493 seconds baseline run down to 362 seconds with TensorRT enabled.

### Q2) Describe what your script does, step by step, from opening a Parquet file to writing an output line.

Pipeline consists of three components: Main Process, Decode Worker Processes and GPU Process with these three sharing memory slots. It starts by reading Parquet file, it reads row by row with its mp4 size, picks the longest video, start the decoder processes based on CPU count of machine less 2, the decoder process, decodes every frame of video from the end and keep them in a separate buffer. For each second, adaptive strategy looks for 2 frames spaced half apart in the second, so we pick only those frames perform BGR+rotation and save in the memory slots allocated for each video process. GPU collects those frames and, group them into batch and provide detections based on criteria defined. The Main process receives the per-second results and writes them per completed video and also watch if any worker dies and it starts a new one.

### Q3) Which part of your final version takes the most time? How did you measure this?

GPU inference is the part that takes most of the time about 95% of the time. I measured it using each process accumulate time per stage and the main process prints the summary of results.

### Q4) How did you check that your results stayed correct after you made the script faster?

Using setup of reference check of 66 videos (2 files of Parquet about 5.2 hours content). Had the every-frame reference for these videos. Each Speed-up tested against the reference to make sure that the overall results stay above the threshold.

### Q5) Describe one suggestion from an AI tool that was wrong or that you decided not to use. How did you find out it was wrong, or why did you reject it?

Not in particular, since we were planning, discussing and writing spec files before even going to write approach. However, it was going to make one silly mistake of creating another repo of an already existing repo on GitHub. I corrected it and pushed back on it since it was not needed.

### Q6) If you had one more week, what would you change, and why?

I check couple of mp4 videos by converting them from npz to mp4. After analysis I found out that the yolov11l model was clearly struggling to capture every object consistently on each second. It needs fine-tuning as well as pre-processing of videos is required before passing to model. But since the grader won't be doing this right now, so I kept this as it is.