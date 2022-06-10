from distutils.core import setup
setup(
  name = 'beewalkvideotrack',
  packages = ['beewalkvideotrack'],
  version = '1.01',
  description = 'In a top down view of a bee, tracks its location.',
  author = 'Mike Smith',
  author_email = 'm.t.smith@sheffield.ac.uk',
  url = 'https://github.com/lionfish0/beewalkvideotrack.git',
  download_url = 'https://github.com/lionfish0/beewalkvideotrack.git',
  keywords = ['image processing','bee','video'],
  classifiers = [],
  install_requires=['numpy','opencv-python'],
  scripts=['bin/beetrack'],
)
