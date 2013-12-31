#!/bin/sh
set -ex

# this can all be replaced with:
# apt-get install libpython2.7-dev:i386
# CC="gcc -m32" LDSHARED="gcc -m32 -shared" LDFLAGS="-m32 -shared" linux32 python setup.py build
# when travis updates to ubuntu 14.04

# setup env
if [ -r /usr/lib/libeatmydata/libeatmydata.so ]; then
  # much faster package installation
  export LD_PRELOAD=/usr/lib/libeatmydata/libeatmydata.so
fi

if [ "$USE_CHROOT" != "1" ] && [ "$USE_BENTO" != "1" ]; then
  # We used to use 'setup.py install' here, but that has the terrible
  # behaviour that if a copy of the package is already installed in
  # the install location, then the new copy just gets dropped on top
  # of it. Travis typically has a stable numpy release pre-installed,
  # and if we don't remove it, then we can accidentally end up
  # e.g. running old test modules that were in the stable release but
  # have been removed from master. (See gh-2765, gh-2768.)  Using 'pip
  # install' also has the advantage that it tests that numpy is 'pip
  # install' compatible, see e.g. gh-2766...
#  pip install .
python setup.py build_ext --inplace
fi

if [ -n "$USE_CHROOT" ] && [ $# -eq 0 ]; then
  sudo apt-get -qq -y --force-yes install debootstrap eatmydata
  DIR=/chroot
  sudo debootstrap --variant=buildd --include=fakeroot,build-essential --arch=i386 --foreign saucy $DIR
  sudo chroot $DIR ./debootstrap/debootstrap --second-stage
  sudo rsync -a $TRAVIS_BUILD_DIR $DIR/
  echo deb http://archive.ubuntu.com/ubuntu/ saucy main restricted universe multiverse | sudo tee -a $DIR/etc/apt/sources.list
  echo deb http://archive.ubuntu.com/ubuntu/ saucy-updates main restricted universe multiverse | sudo tee -a $DIR/etc/apt/sources.list
  echo deb http://security.ubuntu.com/ubuntu saucy-security  main restricted universe multiverse | sudo tee -a $DIR/etc/apt/sources.list
  sudo chroot $DIR bash -c "apt-get update"
  sudo chroot $DIR bash -c "apt-get install -qq -y --force-yes eatmydata libatlas-dev libatlas-base-dev gfortran python-dev python-nose"

  sudo chroot $DIR bash -c "cd numpy && ./.travis.sh chroot-run"
elif [ -n "$USE_BENTO" ] && [ $# -eq 0 ]; then
  export CI_ROOT=$PWD
  cd ..
  
  # Waf
  wget http://waf.googlecode.com/files/waf-1.7.13.tar.bz2
  tar xjvf waf-1.7.13.tar.bz2
  cd waf-1.7.13
  python waf-light
  export WAFDIR=$PWD
  cd ..
  
  # Bento
  wget https://github.com/cournape/Bento/archive/master.zip
  unzip master.zip
  cd Bento-master
  python bootstrap.py
  export BENTO_ROOT=$PWD
  cd ..
  
  cd $CI_ROOT
  
  # In-place numpy build
  $BENTO_ROOT/bentomaker build -i -j
  # Prepend to PYTHONPATH so tests can be run
  export PYTHONPATH=$PWD:$PYTHONPATH
  ./.travis.sh test
else
  # We change directories to make sure that python won't find the copy
  # of numpy in the source directory.
  export PYTHONPATH=$PWD:$PYTHONPATH
  mkdir empty
  cd empty
  INSTALLDIR=$(python -c "import os; import numpy; print(os.path.dirname(numpy.__file__))")
  export PYTHONWARNINGS=default
  python ../tools/test-installed-numpy.py # --mode=full
  # - coverage run --source=$INSTALLDIR --rcfile=../.coveragerc $(which python) ../tools/test-installed-numpy.py
  # - coverage report --rcfile=../.coveragerc --show-missing
#  python setup.py build_ext --inplace
#  python -c "import numpy; numpy.test()"
fi

