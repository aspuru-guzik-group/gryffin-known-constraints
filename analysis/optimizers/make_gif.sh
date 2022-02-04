#!/bin/bash

mkdir tmp
sips -s format gif ./gif_adam*png --out tmp
cd tmp
gifsicle --loop --colors 256 *.gif > gifsicle.gif
cd../
cp tmp/gifsicle.gif .
rm -r tmp
