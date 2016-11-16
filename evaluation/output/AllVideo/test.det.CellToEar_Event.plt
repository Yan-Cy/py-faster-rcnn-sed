set terminal postscript color
set noxzeroaxis
set noyzeroaxis
### Using xScale = log
### Using yScale = nd
set style fill  transparent solid 0.10 noborder
### keyLoc=top keySpacing=0.7 keyFontFace=medium keyFontSize=
set key top samplen 1 spacing 0.7  font "medium,"
set size ratio 0.85
set title 'CellToEar Event'
set grid
set pointsize 3
set ylabel 'PMiss (in %)'
set xlabel 'RFA (in Events/Hour)'

set xtics 
set logscale x
set noytics
set ytics ('5' -1.6449, '10' -1.2816, '20' -0.8416, '40' -0.2533, '60' 0.2533, '80' 0.8416, '90' 1.2816, '95' 1.6449, '98' 2.0537, '99' 2.3263, '99.5' 2.5758, '99.9' 3.0902)
plot [0.01:1000] [-1.64485362793551:3.7190172740417] \
  -x title 'Random Performance' with lines lc 1,\
    'test.det.CellToEar_Event.dat.1' using 5:2 title 'CellToEar Event' with  lines  lc 2