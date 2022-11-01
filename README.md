# beewalkvideotrack
Tracks bee walking route in top down videos

![Example of a frame showing the tracked bee, using the tool](example.png)

# Requirements

For the tracking component you will need:
- numpy
- opencv-python

To use the gettracksummarydataframe method, you will need:
- pandas

For rendering to a file you will also need
- moviepy

Install with, for example:
`pip install moviepy`

# Install

To install this module, run:

```
pip install git+https://github.com/lionfish0/beewalkvideotrack.git
```

# Commandline usage

```
usage: beetrack [-h] [--suffix SUFFIX] [--box BOX] [--smoothtime SMOOTHTIME] [--blur BLUR] [--frames FRAMES] [-r] [--segmentfile SEGMENTFILE] [--mmperpixel MMPERPIXEL] [--combinedfile RECORDFILE] [-s] [-f]
                [-a FORCE_ALL]
                videofn [videofn ...]

Find route bee walks in video(s). Example: beetrack data/*.mp4

positional arguments:
  videofn               Video filename

optional arguments:
  -h, --help            show this help message and exit
  --suffix SUFFIX       For output videos the suffix to use. Default: track.
  --box BOX             Bounding box x1,y1,x2,y2, e.g. 1,2,3,4
  --smoothtime SMOOTHTIME
                        Whether to smooth over time
  --blur BLUR           Whether to smooth over space
  --frames FRAMES       Start and end frame, start,end, e.g. 100,200
  -r                    Whether to output to a video file.
  --segmentfile SEGMENTFILE
                        Whether to create a segmentation CSV file (default True)
  --mmperpixel MMPERPIXEL
                        Resolution of image (mm per pixel). If not included, tries to estimate from a convolution with the patch of squares.
  --combinedfile RECORDFILE
                        CSV file to append with (filename,distance,distancemm) tuple. Default: summarywalkdist.csv
  -s                    Whether to append distance walked to record file.
  -f                    Whether to force a refresh of files already computed (default false, currently checks if the segmentation file exists)
  -a FORCE_ALL          Whether to include files with "track" in their name (default false)
```

## Example commandline

Render to a file (default with suffix '_track')
```
beetrack data/*.mp4 -r
```

Record to a CSV (default name, summarywalkdist.csv)
```
beetrack *.mp4 -s
```

## Python import

To use in your python code, see the [demo notebook](https://github.com/lionfish0/beewalkvideotrack/blob/main/jupyter/Demo.ipynb).

## Description of method from paper

Will include later.

## Citation

To cite this work please use [TBC].

