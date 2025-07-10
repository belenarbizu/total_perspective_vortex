import mne
import argparse

class Report:
    def __init__(self, subject=None, run=None):
        mne.set_log_level("WARNING")

        self.subject, self.run = self.set_numbers(subject, run)
        self.raw = mne.io.read_raw_edf(f"/home/barbizu-/sgoinfre/physionet.org/files/eegmmidb/1.0.0/S{self.subject}/S{self.subject}R{self.run}.edf", preload=True)

        self.raw.pick(picks="eeg").crop(tmax=40).load_data()
        report = mne.Report(title="Raw example")
        report.add_raw(raw=self.raw, title="Raw", psd=False)
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
    #nargs='?' optional arguments
    parser.add_argument("subject", type=int, nargs='?', default=None, help="Subject id")
    parser.add_argument("run", type=int, nargs='?', default=None, help="Run of the subject")
    args = parser.parse_args()
    if args.subject and args.run:
        report = Report(args.subject, args.run)
    else:
        parser.print_help()
        print("Error. You must add [subject] and [run]")


if __name__ == "__main__":
    main()