//  Written by Lucas Olsen for CSCI5105
//  do not modify this file


#ifndef _ML_H_
#define _ML_H_

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>

using namespace std;

///
/////  Matrix functions
///

// Scale a matrix by a scalar value via element wise multiplication
void scale_matricies(vector<vector<double>>& mat, double scalar);

// Sum two matricies together, store the result in _ret
// ASSUMES the two matricies are the same size
void sum_matricies(vector<vector<double>>& _ret, vector<vector<double>>& mat);

// Calculate gradient of two matricies
// curr should be weights after training, orig should be weights before training
// stores result of operation to curr
// ASSUMES the two matricies are the same size
void calc_gradient(vector<vector<double>>& curr, vector<vector<double>>& orig);


///
/////  MLP class - implements a multilayer perceptron
///


class mlp
{
    public:

        mlp();

        bool is_initialized() { return initialized; };

        // Training functions

        // Init model with labeled data, random weights
        bool init_training_random(string fname, int _k, int _h);

        // Init model with labeled data and input weights
        bool init_training_model(string fname, vector<vector<double>>& _V, vector<vector<double>>& _W);

        // Train the model, needs to be initialized first
        // Returns error after training
        double train(double eta, int epochs);

        // Run model on validation data
        // Returns error after validation or -1 on error
        double validate(string fname);

        // Make predicions on data
        // ASSUMES data WITH NO LABELS has been read in
        // Returns int vector of predictions - this vector is empty if error
        vector<int> predict(string fname);

        // Weights functions

        // Copies _V and _W to V and W
        // also sets variables h and k
        void set_weights(vector<vector<double>>& _V, vector<vector<double>>& _W);

        // Copies V and W to input references _V and _W
        void get_weights(vector<vector<double>>& _V, vector<vector<double>>& _W);

        // update weights V and W
        // ASSUMES dV and dW are the same size as V and W respectively
        void update_weights(vector<vector<double>>& _dV, vector<vector<double>>& _dW);

        // NEW for PA2
        // Set momentum factor.  Must be between 0 and 1 (default 0.7)
        void set_momentum(double _momentum) { momentum = _momentum; };

    private:

        // Propogation functions

        // Forwards propogation
        // Uses datapoints in _X array to calculate Y and Z
        void forward_propogate(vector<vector<int>>& _X);

        // Backwards propogation
        // Stores calcaulted weights updates in references _dV and _dW
        void backward_propogate(vector<vector<double>>& _dV, vector<vector<double>>& _dW, double eta);

        // Other private functions

        // Read CSV file of data
        // ASSUMES all input datapoints are integers
        // ASSUMES labels come at the end of the line if included
        bool read_data(vector<vector<int>>& _X, vector<int>& _X_labels, string fname, bool labels);

        // Calculate error of the model
        // Tests Predicitons in Y against labels
        // ASSUMES the model has been built (i.e Y is filled out)
        double error_rate(vector<int>& _X_labels);

        // Calculate error of the model
        double error_func();

        // Activation function, Using ReLU
        double activation_func(double x);

        /* 
        Variable explanations:
            N = number of datapoints
            D = number of features of datapoints (number of input units also)
            H = number of hidden units
            K = number of possible outputs
        */

        int n;
        int d;
        int h;
        int k;

        // Value to check if the network has been intialized
        // Does not necessarily mean trained
        bool initialized = false;

        // NEW for PA2
        // Add momentum to training process to speed up convergence
        double momentum = 0.7;

        // Training datapoints matrix
        // N x D matrix
        vector<vector<int>> X;

        // training data labels
        // N length vector
        vector<int> X_labels;

        // predictions matrix, stores last round of predicitons
        // N x K matrix
        vector<vector<double>> Y;

        // Hidden units matrix, stores hidden units in MLP
        // N x H + 1 matrix
        vector<vector<double>> Z;
        
        // Weights between each input unit and hidden unit
        // D + 1 x H matrix (includes bias)
        vector<vector<double>> W;
        
        // Weights between each hidden unit and output unit
        // H + 1 x K matrix (includes bias)
        vector<vector<double>> V;
};


#endif