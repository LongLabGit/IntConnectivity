
class BurstPicker(object):
    def __init__(self, psth, burst_times):
        self.psth = psth
        self.burst_times = burst_times
        self.n_bursts = 0
        self._modify_bursts = False

    def connect(self):
        """
        connect to all the events we need
        keyboard press 'b' to define new burst
        click burst on-/offset
        keyboard press 'q' to quit
        keyboard press 'u' to undo (remove last burst)
        """
        self.cidpress = self.psth.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidkey = self.psth.figure.canvas.mpl_connect('key_press_event', self.on_key)

    def on_press(self, event):
        if not self._modify_bursts:
            return
        self._add_burst(event.xdata)
        if not self.n_bursts % 2:
            self._modify_bursts = False

    def on_key(self, event):
        if event.key == 'b' or event.key == 'B':
            self._modify_bursts = True
            print 'Click to add burst on- and offset'
        if event.key == 'u' or event.key == 'U':
            self._modify_bursts = True
            print 'Removing last burst'
            self._remove_last_burst()
            self._modify_bursts = False
        if event.key == 'q' or event.key == 'Q':
            print 'Finished adding burst times'
            self.disconnect()

    def disconnect(self):
        self.psth.figure.canvas.mpl_disconnect(self.cidpress)
        self.psth.figure.canvas.mpl_disconnect(self.cidkey)

    def _add_burst(self, burst_time):
        self.burst_times.append(burst_time)
        self.n_bursts += 1
        y_min, y_max = self.psth.get_ylim()
        self.psth.plot((burst_time, burst_time), (y_min, y_max), 'k--', linewidth=0.5)
        self.psth.set_ylim((y_min, y_max))
        self.psth.figure.canvas.draw()

    def _remove_last_burst(self):
        if not len(self.burst_times):
            return
        if not self._modify_bursts:
            return
        del self.burst_times[-1]
        del self.burst_times[-1]
        del self.psth.lines[-1]
        del self.psth.lines[-1]
        self.psth.figure.canvas.draw()
