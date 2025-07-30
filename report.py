import os
import mne
import argparse
import matplotlib.pyplot as plt

class Report:
    def __init__(self, subject, run):
        '''
        Initializes the EEG report generator, loads raw data, preprocesses it, and creates evoked responses.
        Args:
            subject (int, optional): Subject ID number. Defaults to None.
            run (int, optional): Run number for the subject. Defaults to None.
        '''
        mne.set_log_level("WARNING")

        self.subject, self.run = self.set_numbers(subject, run)
        self.raw = mne.io.read_raw_edf(f"/home/barbizu-/sgoinfre/physionet.org/files/eegmmidb/1.0.0/S{self.subject}/S{self.subject}R{self.run}.edf", preload=True)

        self.preprocess()

        self.raw_filtered.pick(picks="eeg").crop(tmax=40).load_data()
        self.events, event_id = mne.events_from_annotations(self.raw_filtered)
        self.epochs = mne.Epochs(raw=self.raw_filtered, events=self.events, event_id=event_id, baseline=(None, 0))

        self.evokeds = []
        self.evokeds_title = []
        for cond in event_id.keys():
            self.evokeds.append(self.epochs[cond].average())
            self.evokeds_title.append(f"Evoked: {cond}")

        self.create_html()
    

    def preprocess(self):
        if not os.path.exists("figs"):
                    os.makedirs("figs")

        channels_names = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5',
        'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
        'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3',
        'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2',
        'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10',
        'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4',
        'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz']

        mapping = {name: channels_names[i] for i, name in enumerate(self.raw.ch_names)}
        self.raw.rename_channels(mapping)

        montage = mne.channels.make_standard_montage('standard_1020')
        self.raw.set_montage(montage)
            
        fig = montage.plot(show=False)
        plt.savefig("figs/montage_plt.png")
        plt.close(fig)

        self.raw_filtered = self.raw.copy()
        self.raw_filtered.filter(l_freq=8., h_freq=40., picks='eeg')
        self.raw_filtered.notch_filter(freqs=60)

        fig = self.raw.plot(n_channels=10, duration=5.0, scalings='auto', title="Raw")
        plt.savefig("figs/raw_data.png")
        plt.close(fig)
        
        fig = self.raw_filtered.plot(n_channels=10, duration=5.0, scalings='auto', title="Raw Filtered")
        plt.savefig("figs/raw_filtered.png")
        plt.close(fig)

        self.raw_filtered.set_eeg_reference('average', projection=True)

        fig = self.raw_filtered.plot(n_channels=10, duration=5.0, scalings='auto', title="Raw Reference Change")
        plt.savefig("figs/reference_change.png")
        plt.close(fig)


    def create_html(self):
        report = mne.Report(title=f"Report {self.subject} {self.run}")
        
        report.add_image(image="figs/montage_plt.png", title="Montage")
        report.add_image(image="figs/raw_data.png", title="Raw data", section="Comparation data")
        report.add_image(image="figs/raw_filtered.png", title="Raw filtered", section="Comparation data")
        report.add_image(image="figs/reference_change.png", title="Reference change", section="Comparation data")
        report.add_raw(raw=self.raw, title="Raw", psd=True)
        report.add_raw(raw=self.raw_filtered, title="Raw Filtered", psd=True)
        report.add_events(events=self.events, title="Events", sfreq=self.raw_filtered.info["sfreq"])
        report.add_epochs(epochs=self.epochs, title="Epochs")
        report.add_evokeds(evokeds=self.evokeds, titles=self.evokeds_title)

        report.save("report_raw.html", overwrite=True)

    
    def set_numbers(self, subject, run):
        if subject < 10:
            subject = "00" + str(subject)
        elif subject >= 10 and subject < 100:
            subject = "0" + str(subject)
        if run < 10:
            run = "0" + str(run)
        return subject, run

def main():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument("subject", type=int, choices=range(1, 110), help="Subject id")
    parser.add_argument("run", type=int, choices=range(3, 15), help="Run of the subject")
    args = parser.parse_args()
    report = Report(args.subject, args.run)


if __name__ == "__main__":
    main()