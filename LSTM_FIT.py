class FIT:
    def __init__(self, train_X, train_y, dropout, recurrent_dropout, cell_units,  cell_units_l2, epochs): 
        self.train_X = train_X,
        self.train_y = train_y
        self.cell_units = cell_units
        self.cell_units_l2 = cell_units_l2
        self.epochs = epochs
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
    
    def LSTM(self):
        """
        Fit LSTM to data train_X, train_y 
    
        arguments
        ---------
        train_X (array): input sequence samples for training 
        train_y (list): next step in sequence targets
        cell_units (int): number of neurons in LSTM cells  
        epochs (int): number of training epochs   
        """
    
        # initialize model
        model = Sequential() 
    
        # construct a LSTM layer with specified number of neurons
        # per cell and desired sequence input format 
        model.add(LSTM(self.cell_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, input_shape=(self.train_X.shape[1],1),return_sequences= True))
        model.add(LSTM(self.cell_units_l2))
        # add an output layer to make final predictions 
        model.add(Dense(1))
    
        # define the loss function / optimization strategy, and fit
        # the model with the desired number of passes over the data (epochs) 
        model.compile(loss='mean_squared_error', optimizer='adam')
        if self.train_X.shape[0]%4 == 0: 
            valx, valy = self.train_X[-int(self.train_X.shape[0]/4):], self.train_y[-int(self.train_X.shape[0]/4):] #valx, valy is last quarter of train_X, train_Y 
            history = model.fit(self.train_X, self.train_y, epochs=self.epochs, validation_data = (valx, valy), shuffle = False, batch_size=int(self.train_X.shape[0]/4), verbose=1) #
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model train vs validation loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper right')
            plt.show()
            return model
        else:
            print('input month is not a multiple of 4. Cannot divide into integer batches')
            valx, valy = self.train_X[-int(self.train_X.shape[0]/4):], self.train_y[-int(self.train_X.shape[0]/4):]
            history = model.fit(self.train_X, self.train_y, epochs=self.epochs, validation_data = (valx, valy), shuffle = False, batch_size=int(self.train_X.shape[0]/4), verbose=1) #
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model train vs validation loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper right')
            plt.show()
            return model
        else:
            print('input month is not a multiple of 4. Cannot divide into integer batches')