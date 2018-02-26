from keras.models import Model

class BaseModel(object):
    def __init__(self,model_name="test"):
        self.model_name=model_name

    def fit(self,X_train,Y_train,X_val,Y_val,n_epochs,batch_size,plot_history=False,verbose=1):
        pass


class RNN(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        input = Input(shape=(), dtype='int32', )
        self.model= Model(input, preds)