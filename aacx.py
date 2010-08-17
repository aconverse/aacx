#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as pyplot
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg
import re
from scipy.io import wavfile

import gtk

swb_offset_1024_96 = [
      0,   4,   8,  12,  16,  20,  24,  28,
     32,  36,  40,  44,  48,  52,  56,  64,
     72,  80,  88,  96, 108, 120, 132, 144,
    156, 172, 188, 212, 240, 276, 320, 384,
    448, 512, 576, 640, 704, 768, 832, 896,
    960, 1024
]

swb_offset_128_96 = [
    0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 92, 128
]

swb_offset_1024_64 = [
      0,   4,   8,  12,  16,  20,  24,  28,
     32,  36,  40,  44,  48,  52,  56,  64,
     72,  80,  88, 100, 112, 124, 140, 156,
    172, 192, 216, 240, 268, 304, 344, 384,
    424, 464, 504, 544, 584, 624, 664, 704,
    744, 784, 824, 864, 904, 944, 984, 1024
]

swb_offset_1024_48 = [
      0,   4,   8,  12,  16,  20,  24,  28,
     32,  36,  40,  48,  56,  64,  72,  80,
     88,  96, 108, 120, 132, 144, 160, 176,
    196, 216, 240, 264, 292, 320, 352, 384,
    416, 448, 480, 512, 544, 576, 608, 640,
    672, 704, 736, 768, 800, 832, 864, 896,
    928, 1024
]

swb_offset_128_48 = [
     0,   4,   8,  12,  16,  20,  28,  36,
    44,  56,  68,  80,  96, 112, 128
]

swb_offset_1024_32 = [
      0,   4,   8,  12,  16,  20,  24,  28,
     32,  36,  40,  48,  56,  64,  72,  80,
     88,  96, 108, 120, 132, 144, 160, 176,
    196, 216, 240, 264, 292, 320, 352, 384,
    416, 448, 480, 512, 544, 576, 608, 640,
    672, 704, 736, 768, 800, 832, 864, 896,
    928, 960, 992, 1024
]

swb_offset_1024_24 = [
      0,   4,   8,  12,  16,  20,  24,  28,
     32,  36,  40,  44,  52,  60,  68,  76,
     84,  92, 100, 108, 116, 124, 136, 148,
    160, 172, 188, 204, 220, 240, 260, 284,
    308, 336, 364, 396, 432, 468, 508, 552,
    600, 652, 704, 768, 832, 896, 960, 1024
]

swb_offset_128_24 = [
     0,   4,   8,  12,  16,  20,  24,  28,
    36,  44,  52,  64,  76,  92, 108, 128
]

swb_offset_1024_16 = [
      0,   8,  16,  24,  32,  40,  48,  56,
     64,  72,  80,  88, 100, 112, 124, 136,
    148, 160, 172, 184, 196, 212, 228, 244,
    260, 280, 300, 320, 344, 368, 396, 424,
    456, 492, 532, 572, 616, 664, 716, 772,
    832, 896, 960, 1024
]

swb_offset_128_16 = [
     0,   4,   8,  12,  16,  20,  24,  28,
    32,  40,  48,  60,  72,  88, 108, 128
]

swb_offset_1024_8 = [
      0,  12,  24,  36,  48,  60,  72,  84,
     96, 108, 120, 132, 144, 156, 172, 188,
    204, 220, 236, 252, 268, 288, 308, 328,
    348, 372, 396, 420, 448, 476, 508, 544,
    580, 620, 664, 712, 764, 820, 880, 944,
    1024
]

swb_offset_128_8 = [
     0,   4,   8,  12,  16,  20,  24,  28,
    36,  44,  52,  60,  72,  88, 108, 128
]

swb_offset_1024 = [
    swb_offset_1024_96, swb_offset_1024_96, swb_offset_1024_64,
    swb_offset_1024_48, swb_offset_1024_48, swb_offset_1024_32,
    swb_offset_1024_24, swb_offset_1024_24, swb_offset_1024_16,
    swb_offset_1024_16, swb_offset_1024_16, swb_offset_1024_8,
    swb_offset_1024_8
]

swb_offset_128 = [
    swb_offset_128_96, swb_offset_128_96, swb_offset_128_96,
    swb_offset_128_48, swb_offset_128_48, swb_offset_128_48,
    swb_offset_128_24, swb_offset_128_24, swb_offset_128_16,
    swb_offset_128_16, swb_offset_128_16, swb_offset_128_8,
    swb_offset_128_8
]

window_sequence_name = [
	"ONLY_LONG_SEQUENCE",
	"LONG_START_SEQUENCE",
	"EIGHT_SHORT_SEQUENCE",
	"LONG_STOP_SEQUENCE"
]

window_shape_name = [
	"SIN",
	"KBD",
]

def mk_kbd_window(alpha, N):
	n = numpy.arange(0., N/2)
	n = (n - N/4)/(N/4)
	n = n * n
	W = numpy.i0(numpy.pi*alpha*numpy.sqrt(1.0 - n))
	W = numpy.cumsum(W)
	W = numpy.sqrt(W / W[-1])
	return W

def mk_sin_window(N):
	return numpy.sin(numpy.pi / N * (numpy.arange(0., N/2) + 0.5))

kbd_1024 = mk_kbd_window(4, 2048)
kbd_128  = mk_kbd_window(6, 256)
sin_1024 = mk_sin_window(2048)
sin_128  = mk_sin_window(256)

window_group_colors = [
	'#FFC200', '#FF5B00', '#84002E', '#4AC0F2', '#FFC200', '#FF5B00', '#84002E', '#4AC0F2'
]

class Frame: 
	def __init__(self):
		self.global_gain = 0;
		self.window_sequence = 0;
		self.window_shape = 0;
		self.window_shape_prev = 0;
		self.common_window = 0;
		self.sampling_index = 0;
		self.group_len = list();
		self.tns = 0;
		self.sf = numpy.empty([0], dtype='int32')
		self.mdct = numpy.empty([0], dtype='float')

def read_fflog(logname, wavname):
	print 'begin'
	sfs = numpy.empty([0], dtype='int32')
	fd = open(logname, 'r')
	line_it = iter(fd.readlines())
	fd.close()
	frame = Frame()
	frames = list()
	line = ""
	try:
		print "first loop"
		while line_it:
			line = line_it.next()
			if line.rstrip() == "Press [q] to stop encoding":
				break;
		print "second loop"
		while line_it:
			line = line_it.next()
			m = re.match(r"\[aac.*\] frame (\d+)", line)
			if m:
				print "frame", m.group(1)
				continue
			m = re.match(r"\[aac.*\] sampling_index (\d)", line)
			if m:
				frame.sampling_index = int(m.group(1))
				continue
			m = re.match(r"\[aac.*\] common_window (\d)", line)
			if m:
				frame.common_window = int(m.group(1))
				continue
			m = re.match(r"\[aac.*\] window_shape (\d)", line)
			if m:
				frame.window_shape = int(m.group(1))
				continue
			m = re.match(r"\[aac.*\] window_shape_prev (\d)", line)
			if m:
				frame.window_shape_prev = int(m.group(1))
				continue
			m = re.match(r"\[aac.*\] window_sequence (\d)", line)
			if m:
				frame.window_sequence = int(m.group(1))
				continue
			m = re.match(r"\[aac.*\] group_len (\d): (\d)", line)
			if m:
				frame.group_len.append(int(m.group(2)))
				continue
			m = re.match(r"\[aac.*\] tns_present (\d)", line)
			if m:
				frame.tns = int(m.group(1))
				continue
			m = re.match(r"\[aac.*\] mdct coef +\d+: (\S+) (\S+) (\S+) (\S+)", line)
			if m:
				frame.mdct = numpy.append(frame.mdct, float(m.group(1)))
				frame.mdct = numpy.append(frame.mdct, float(m.group(2)))
				frame.mdct = numpy.append(frame.mdct, float(m.group(3)))
				frame.mdct = numpy.append(frame.mdct, float(m.group(4)))
				continue
			m = re.match(r"\[aac.*\] scalefactors begin global_gain *(\d+) max_sfb *(\d+)", line)
			if m:
				frame.global_gain = int(m.group(1))
				frame.max_sfb = int(m.group(2))
				while line_it:
					line = line_it.next()
					m = re.match(r"\[aac.*\]  sf *(\d+): *(\d+)", line)
					if m:
						frame.sf = numpy.append(frame.sf, numpy.int32(m.group(2)))
						continue
					if re.match(r"\[aac.*\] scalefactors end", line):
						break
				continue
			if re.match(r"\[aac.*\] frame end", line):
				frames.append(frame)
				frame = Frame()
	except StopIteration, e:
		e = 0
	print "processed", len(frames), "frames"
	wav = wavfile.read(wavname)
	print wav[1].shape
	wav = numpy.append(wav[1], numpy.zeros(1024, dtype='int16'))
	return (wav, frames)

def aac_show_frame(fig, stuff, n):
	wav = stuff[0]
	frame = stuff[1][n]
	global_gain = frame.global_gain
	fsf = frame.sf
	sfx = numpy.empty([0], dtype='int32')
	max_sfb = frame.max_sfb
	sri = frame.sampling_index
	if frame.window_sequence != 2:
		for i in range(0, max_sfb):
			sfx = numpy.append(sfx, numpy.repeat(fsf[i], swb_offset_1024[sri][i+1]-swb_offset_1024[sri][i]))
		sfx = numpy.append(sfx, numpy.zeros(1024-sfx.shape[0], dtype='int32'))
		color_idx = numpy.zeros(8, dtype='int32')
	else:
		groups = len(frame.group_len)
		color_idx = numpy.empty([0], dtype='int32')
		g_idx = 0
		for g in frame.group_len:
			sf = numpy.empty([0], dtype='int32')
			for i in range(0, max_sfb):
				sf = numpy.append(sf, numpy.repeat(fsf[g_idx*max_sfb+i], swb_offset_128[sri][i+1]-swb_offset_128[sri][i]))
			sf = numpy.append(sf, numpy.zeros(128-sf.shape[0], dtype='int32'))
			sfx = numpy.append(sfx, numpy.tile(sf, g))
			color_idx = numpy.append(color_idx, numpy.tile(g_idx, g))
			g_idx = g_idx + 1

	b = numpy.arange(0, 1024)
	t = n*1024 + numpy.arange(0, 2048)
	fig.clf()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	ax3 = ax2.twinx()
	ax4 = ax1.twinx()
	if frame.window_sequence == 2:
		group_edges = numpy.cumsum(frame.group_len)
		win_l = kbd_128 if frame.window_shape_prev else sin_128
		win_r = kbd_128 if frame.window_shape      else sin_128
		win = numpy.append(win_l, win_r[::-1])
		g = 0
		ax4.plot(t[0] + 448 + numpy.arange(0, 256), win, window_group_colors[g])
		win = numpy.append(win_r, win_r[::-1])
		for i in range(1, 8):
			if i >= group_edges[g]:
				g = g + 1
			ax4.plot(t[0] + 448 + i*128 + numpy.arange(0, 256), win, window_group_colors[g])
	else:
		if frame.window_sequence == 3:
			win_l = numpy.zeros(448, dtype='float')
			win_l = numpy.append(win_l, kbd_128 if frame.window_shape_prev else sin_128)
			win_l = numpy.append(win_l, numpy.ones(448, dtype='float'))
		else:
			win_l = kbd_1024 if frame.window_shape_prev else sin_1024
		if frame.window_sequence == 1:
			win_r = numpy.zeros(448, dtype='float')
			win_r = numpy.append(win_r, kbd_128 if frame.window_shape else sin_128)
			win_r = numpy.append(win_r, numpy.ones(448, dtype='float'))
		else:
			win_r = kbd_1024 if frame.window_shape else sin_1024
		win = numpy.append(win_l, win_r[::-1])
		ax4.plot(t, win, window_group_colors[0])
	ax4.axis([t[0], t[-1]+1, 0, 1.005]) #hack to keep the top from diappearing
	ax4.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
	ax4.xaxis.set_major_locator(matplotlib.ticker.LinearLocator(9))
	ax1.plot(t, wav[t], 'b')
	ax1.axis([t[0], t[-1]+1, -32768, 32767])
	lcoef = 20*numpy.log10(numpy.abs(frame.mdct)+1)
	#ax2.fill_between(t, lcoef, 0, linewidth=0.1)
	#ax2.step(t, lcoef)
	ax2.plot(numpy.arange(0, 1024), lcoef)
	g = numpy.tile(global_gain, 1024)
	ax3.step(b, g, 'r')
	if frame.window_sequence == 2:
		for i in range(0, 8):
			x = i*128 + numpy.arange(0, 128)
			ax3.step(x, sfx[x], window_group_colors[color_idx[i]])
	else:
		ax3.step(b, sfx[b], window_group_colors[color_idx[0]])
	ax3.axis([b[0], b[-1]+1, 0, 255])
	ax3.xaxis.set_major_locator(matplotlib.ticker.LinearLocator(9))
	ax3.set_ylabel('global_gain', color='r')
	return (fig, ax1, ax2)

class aacxGui:
	def __init__(self):
		self.builder = gtk.Builder()
		self.builder.add_from_file("aacx.ui")
		self.window = self.builder.get_object("window1")
		self.status = self.builder.get_object("statusbar1")
		self.frmadj = self.builder.get_object("frame_adj")
		self.about  = gtk.AboutDialog()

		self.about.set_name("AACX")
		self.about.set_version("0.0.1")
		self.about.set_copyright(u'Copyright © 2010 Alex Converse')
		self.about.set_website("http://github.com/aconverse/aacx")
		self.about.set_license("GPLv2+")

		self.window.connect("destroy", gtk.main_quit)
		signals = {
			"on_file_quit_activate": gtk.main_quit,
			"on_help_about_activate": self.show_about,
			"on_frame_adj_value_changed": self.spinback,
		}
		self.builder.connect_signals(signals)
		self.set_status_here("xxx")
		self.hasplot = 0
	def addplot(self, fig):
		self.canvas = FigureCanvasGTKAgg(fig)
		self.canvas.show()
		v = self.builder.get_object("vbox1")
		v.pack_start(self.canvas)
		self.hasplot = 1
	def redraw(self):
		self.canvas.draw()
	def show(self):
		self.window.show()
	def show_about(self, data):
		self.about.run()
		self.about.hide()
	def set_status_here(self, text):
		self.status.push(0, text)
	def spinback(self, data):
		spinback(int(data.get_value()), int(data.get_upper())+1)
	def set_spinner(self, n):
		self.frmadj.set_value(n)
	def set_num_frames(self, n):
		self.frmadj.set_upper(n-1)

stuff = read_fflog("fflog", "out.wav")
num_frames = len(stuff[1])
x = aacxGui()
x.set_num_frames(num_frames)
fig = pyplot.figure()

def spinback(n, num_frames):
	text =  "frame "+str(n)+"/"+str(num_frames)+" | "+window_sequence_name[stuff[1][n].window_sequence]
	text += " | "+window_shape_name[stuff[1][n].window_shape_prev]+"/"+window_shape_name[stuff[1][n].window_shape_prev]
	text += " | "+" global gain "+str(stuff[1][n].global_gain)
	x.set_status_here(text)
	aac_show_frame(fig, stuff, n)
	if x.hasplot:
		x.redraw()
	else:
		x.addplot(fig)
	x.set_spinner(n)

spinback(1, num_frames)
x.show()
gtk.main()
