import mne
import matplotlib.pyplot as plt
import numpy as np
import argparse
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, ShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pca import PCA
from csp import CSP

param_grid_lda = {
    "pca__n_components": [2, 4, 6, 8, 10],
    "lda__store_covariance": [True, False],
    "lda__solver": ['svd', 'lsqr']
}

BAD_SUBJECTS = [100, 104, 106]

class Data_EEG:
    def __init__(self, subject=None, run=None, task=None):
        mne.set_log_level("WARNING")
        if not subject and not run and not task:
            self.experiments()
        else:
            self.subject, self.run = self.set_numbers(subject, run)
            self.raw = mne.io.read_raw_edf(f"/home/barbizu-/sgoinfre/physionet.org/files/eegmmidb/1.0.0/S{self.subject}/S{self.subject}R{self.run}.edf", preload=True)

            #change channels names
            self.change_channels_names()
            
            #set montage to standard_1020
            montage = mne.channels.make_standard_montage('standard_1020')
            self.raw.set_montage(montage)

            #filter data
            self.raw_filtered = self.raw.copy()
            self.raw_filtered.filter(l_freq=8., h_freq=40., picks='eeg')
            self.raw_filtered.notch_filter(freqs=60)
            
            #set reference
            self.raw_filtered.set_eeg_reference('average', projection=True)

            self.events, self.event_id = mne.events_from_annotations(self.raw_filtered)

            epochs, power = self.epoch_data()
            X = power.get_data(picks="eeg")
            X = X.mean(axis=-1).reshape(len(X), -1)
            y = epochs.events[:, 2]
            self.split_datasets(X, y)

            if task == "train":
                self.train(epochs, "train")
                if not os.path.exists("models"):
                    os.makedirs("models")
                joblib.dump(self.grid, f"models/model_{self.subject}_{self.run}.pkl")
            elif task == "predict":
                self.predict("train")


    def change_channels_names(self):
        channels_names = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5',
        'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
        'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3',
        'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2',
        'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10',
        'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4',
        'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz']

        mapping = {name: channels_names[i] for i, name in enumerate(self.raw.ch_names)}
        self.raw.rename_channels(mapping)


    def epoch_data(self):
        channels_to_use = ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4']
        picks = mne.pick_channels(self.raw_filtered.info['ch_names'], include=channels_to_use)
        epochs = mne.Epochs(self.raw_filtered, self.events, event_id=self.event_id, picks=picks, tmin=-0.1, tmax=4.0, baseline=None, preload=True, verbose=False)
        epochs.drop_bad()
        epochs = epochs["T1", "T2"]

        freqs = np.arange(8, 40, 2) #frequencias desde 8 a 40 Hz
        power = epochs.compute_tfr('morlet', tmin=-0.1, tmax=4.0, freqs=freqs, return_itc=False)

        return epochs, power


    def set_numbers(self, subject, run):
        if subject < 10:
            subject = "00" + str(subject)
        elif subject >= 10 and subject < 100:
            subject = "0" + str(subject)
        if run < 10:
            run = "0" + str(run)
        return subject, run


    def split_datasets(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


    def train(self, epochs, task="experiment"):
        self.pipe_lda = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('lda', LinearDiscriminantAnalysis())
        ])

        cv = ShuffleSplit(10, test_size=0.2, random_state=42)
        scores = cross_val_score(self.pipe_lda, self.X_train, self.y_train, cv=cv)
        if not task == "experiment":
            print(scores)
            print("cross_val_score:", scores.mean())
        self.grid = GridSearchCV(self.pipe_lda, param_grid_lda, cv=cv, n_jobs=-1, scoring="accuracy", verbose=False)
        self.grid.fit(self.X_train, self.y_train)
    

    def predict(self, task="experiment"):
        if not task == "experiment":
            try:
                self.grid = joblib.load(f"models/model_{self.subject}_{self.run}.pkl")
            except Exception as e:
                print(e)
                exit(1)

        accuracy = []
        for n in range(self.X_test.shape[0]):
            pred = self.grid.best_estimator_.predict(self.X_test[n].reshape(1, -1))[0]
            truth = self.y_test[n:n + 1][0]
            accuracy.append(pred == truth)
            if not task == "experiment":
                self.print_pred(0, pred, truth)
        mean_accuracy = np.mean(accuracy)
        if not task == "experiment":
            print("Accuracy: ", mean_accuracy)
        return mean_accuracy


    def print_pred(self, epoch, pred_val, true_val):
        result = True if pred_val == true_val else False
        print(f"epoch {epoch}: {pred_val} {true_val} {result}")


    def experiments(self):
        #experiment 0, [3, 7, 11] T1 vs T2, left vs right fist
        #experiment 1, [5, 9, 13] T1 vs T2, both fists vs both feet
        #experiment 2, [4, 8, 12] T1 vs T2, imagine left vs right fist
        #experiment 3, [6, 10, 14] T1 vs T2, imagine both fists vs both feet
        #experiment 4, [4, 8, 12] T0 vs T1, rest vs left fist
        #experiment 5, [5, 9, 13] T0 vs T2, rest vs both feet

        subjects = [n for n in range(1, 110) if n not in BAD_SUBJECTS]

        ranges = {
            0 : [3, 12],
            1 : [5, 14],
            2 : [4, 13],
            3 : [6, 15],
            4 : [4, 13],
            5 : [5, 14]
        }
        events = {
            0 : ["T1", "T2"],
            1 : ["T1", "T2"],
            2 : ["T1", "T2"],
            3 : ["T1", "T2"],
            4 : ["T0", "T1"],
            5 : ["T0", "T2"]
        }

        accuracy = []
        total_accuracy = 0
        print("Mean accuracy of the six different experiments for all 109 subjects:")
        for i in range(0, 6):
            accuracy.append(self.experiment_n(i, subjects, ranges[i], events[i]))
        for i in range(0, 6):
            print(f"experiment {i}:    accuracy = {accuracy[i]}")
            total_accuracy += accuracy[i]

        print(f"Mean accuracy of 6 experiments: {total_accuracy / 6}")

    
    def experiment_n(self, num, subjects, ranges, events):
        channels_to_use = ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4']
        total_accuracy = 0
        for subject in subjects:
            subject_accuracy = 0
            for run in range(ranges[0], ranges[1], 4):
                self.subject, self.run = self.set_numbers(subject, run)
                self.raw = mne.io.read_raw_edf(f"/sgoinfre/students/barbizu-/physionet.org/files/eegmmidb/1.0.0/S{self.subject}/S{self.subject}R{self.run}.edf", preload=True)
                self.change_channels_names()
                self.raw_filtered = self.raw.copy().filter(l_freq=8., h_freq=40.)
                self.raw_filtered.set_eeg_reference('average', projection=True)
                picks = mne.pick_channels(self.raw_filtered.info['ch_names'], include=channels_to_use)
                self.events, self.event_id = mne.events_from_annotations(self.raw_filtered)
                epochs = mne.Epochs(self.raw_filtered, self.events, event_id=self.event_id, picks=picks, tmin=0, tmax=4.0, baseline=None, preload=True, verbose=False)
                epochs.drop_bad()
                epochs = epochs[events]
                freqs = np.arange(8, 40, 2) #frequencias desde 8 a 40 Hz
                power = epochs.compute_tfr('morlet', tmin=-0.1, tmax=4.0, freqs=freqs, return_itc=False)
                X = power.get_data(picks="eeg")
                X = X.mean(axis=-1).reshape(len(X), -1)
                y = epochs.events[:, 2]
                self.split_datasets(X, y)
                self.train(epochs)
                accuracy = self.predict()
                total_accuracy += accuracy
                subject_accuracy += accuracy
            subject_accuracy = subject_accuracy / 3
            print(f"experiment {num}: subject {self.subject}: accuracy = {subject_accuracy}")
        return (total_accuracy / (len(subjects) * 3))


def main():
    parser = argparse.ArgumentParser(description='.')
    #nargs='?' optional arguments
    parser.add_argument("subject", type=int, nargs='?', default=None, help="Subject id")
    parser.add_argument("run", type=int, nargs='?', default=None, help="Run of the subject")
    parser.add_argument("task", choices=["train", "predict"], nargs='?', default=None, help="Task to perform")
    args = parser.parse_args()
    if args.subject and args.run and args.task:
        data = Data_EEG(args.subject, args.run, args.task)
    elif not args.subject and not args.run and not args.task:
        data = Data_EEG()
    else:
        parser.print_help()
        print("Error. You must add [subject], [run], [{train,predict}]")


if __name__ == "__main__":
    main()