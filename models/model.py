class Model:
    
    def load_data(self):
        """ Load data from file.
        """
        pass

    def add_placeholders(self):
        """ Create placeholder variables.
        """
        pass

    def add_model(self):
        """ Add Tensorflow ops to get scores from inputs.
        """
        pass

    def add_loss_op(self):
        """ Add Tensorflow op to compute loss.
        """
        pass

    def add_training_op(self):
        """ Add Tensorflow op to run one iteration of SGD.
        """
        pass

    def add_pred_op(self):
        """ Add Tensorflow op to generate predictions.
        """
        pass

    def _train(self):
        """ Train on an image and return pertinent data.
        """
        pass

    def _validation(self):
        """ Validate on an image and return pertinent data.
        """
        pass

    def _segment(self):
        """ Segment an image and return pertinent data.
        """
        pass
