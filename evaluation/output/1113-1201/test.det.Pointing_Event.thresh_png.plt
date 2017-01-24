## GNUPLOT command file
set terminal png truecolor medium size 800,800 crop
set style data lines
set title 'Threshold Plot for Pointing Event'
set xlabel 'Detection Score'
set grid
set size ratio 0.85
plot [0.106000:0.991000]  \
  'test.det.Pointing_Event.dat.1' using 1:4 title 'PMiss' with lines lt 2, \
  'test.det.Pointing_Event.dat.1' using 1:5 title 'RFA' with lines lt 3, \
  'test.det.Pointing_Event.dat.1' using 1:6 title 'DCR' with lines lt 4, \
  'test.det.Pointing_Event.dat.2' using 1:2 title 'Min DCR 0.949, scr 0.244' with points lt 6
