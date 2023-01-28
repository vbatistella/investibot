import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

FILES_DIRECTORY = "../dataset/Stocks/"
MEAN = 28.08
STD = 78.115
WINDOW_SIZE = 200

class Data:
    def __init__(self, batch_size, input, output, shift):
        self.batch_size = batch_size
        self.input      = input
        self.output     = output
        self.shift      = shift

    # Get batch of data
    def get_batch(self, i):
        print("Batch:", i)

        # Get files that will be used on batch
        # Return [] if done
        print("Getting filenames")
        file_list = self.get_file_names(i)
        if not file_list:
            print("Done batching")
            return np.array([])
        
        # Filter files
        print("Filtering files...")
        file_list = self.filter_files(file_list)
        print("Processing on [", len(file_list), "] files")
        print("Loading files...")
        # Get matrix of data from files
        raw_data = self.get_data_from_files(file_list)
        # Normalize roughly between 0 and 1
        print("Normalizing data...")
        normalized = self.normalize(raw_data)
        # Moving average
        print("Preprocessing...")
        preprocessed = self.pre_process(normalized, WINDOW_SIZE)
        # Prepare data for model
        print("Formating...")
        x, y = self.format(preprocessed)
        print("Splitting...")
        x_train, y_train, x_test, y_test = self.split_dataset(x, y, 0.25)
        print(x_train.shape)
        return x_train, y_train, x_test, y_test


    # Get filenames from a batch
    # def get_file_names(self, i):
    #     dire = FILES_DIRECTORY
    #     all_files = os.listdir(dire)
    #     return all_files[i*self.batch_size:(i+1)*self.batch_size]
    
    def get_file_names(self, filename):
        with open(filename) as f:
            lines = [line.rstrip('\n') for line in f]
        return lines
    
    def filter_files(self, file_list):
        filtered = []
        for f in file_list:
            try:
                df = self.open(os.path.join(FILES_DIRECTORY, f))
            except:
                continue
            if len(df["Close"].tolist()) >= self.input+self.output+self.shift:
                filtered.append(f)
        return filtered
    
    def get_data_from_files(self, file_list):
        data_raw = []
        for f in file_list:
            df = self.open(os.path.join(FILES_DIRECTORY, f))
            df = df["Close"].tolist()
            data_raw.append(df)
        return data_raw

    def open(self, filename):
        df = pd.read_csv(filename)
        return df
    
    # def normalize(self, data):
    #     normalized = []
    #     for i in data:
    #         aux = np.array(i)
    #         aux = (aux - MEAN) / STD
    #         normalized.append(aux.tolist())
        
    #     return normalized

    def normalize(self, data):
        scaler = MinMaxScaler(feature_range=(0,1))
        data2 = np.array(data).reshape((len(data[0]),1))
        normalized = scaler.fit_transform(data2)
        normalized = normalized.reshape(len(data[0]))
        normalized = normalized.tolist()
        return [normalized]
    
    def pre_process(self, data, window):
        preproc = []
        for j in range(len(data)):
            preproc.append([])
            for i in range(window, len(data[j])):
                m = sum(data[j][i-window:i])/window
                preproc[j].append(m)
        return preproc
    
    def format(self, data):
        x = np.array([])
        y = np.array([])

        for i in range(len(data)):
            x_a, y_a = self.batch(data[i])
            x = np.append(x, x_a)
            y = np.append(y, y_a)
        
        x = x.reshape(x.size//self.input, self.input, 1)
        y = y.reshape(y.size//self.output, self.output, 1)

        return x, y
    
    def batch(self, data):
        x = np.array([])
        y = np.array([])

        s = len(data)-(self.shift+self.output+self.input)
        for i in range(s):
            x_step = data[i:i+self.input]
            y_step = data[i+self.input+self.shift:i+self.input+self.shift+self.output]
            
            x = np.append(x, x_step)
            y = np.append(y, y_step)

        x = x.reshape((s,self.input))
        y = y.reshape((s,self.output))
        
        return x, y

    def split_dataset(self, x, y, test_percentage):
        test_split = int(x.shape[0]*test_percentage)
        # train
        x_train = x[:-test_split]
        y_train = y[:-test_split]
        # test
        x_test = x[-test_split:]
        y_test = y[-test_split:]
        return x_train, y_train, x_test, y_test

    def show(self, model, file):
        raw_data = self.get_data_from_files([file])
        y = np.array(self.normalize(raw_data))
        y = y.reshape(y.size)
        x = np.array(range(y.size))

        plt.plot(x[self.input+self.shift:], y[self.input+self.shift:])

        prediction = []
        for i in range(0, y.size-self.input-self.shift, self.output):
            inp = np.array(y[i:i+self.input])
            p = model.infer(inp.reshape(1,self.input,1))[0][0]
            prediction.append(p)

        plt.plot(x[self.input+self.shift:], prediction)
        plt.show()