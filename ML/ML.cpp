//  Written by Lucas Olsen for CSCI5105
//  Do not modify this file


#include "ML.hpp"

using namespace std;


///
/////
/////  Matrix helper functions
/////
///


// Scale a matrix by a scalar value via element wise multiplication
void scale_matricies(vector<vector<double>>& mat, double scalar)
{
    int rows = mat.size();
    int cols = mat[0].size();
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
            mat[i][j] *= scalar;
}

// Sum two matricies together, store the result in _ret
// ASSUMES the two matricies are the same size
void sum_matricies(vector<vector<double>>& _ret, vector<vector<double>>& mat)
{
    int rows = mat.size();
    int cols = mat[0].size();
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
            _ret[i][j] += mat[i][j];
}

// Calculate gradient of two matricies
// curr should be weights after training, orig should be weights before training
// stores result of operation to curr
// ASSUMES the two matricies are the same size
void calc_gradient(vector<vector<double>>& curr, vector<vector<double>>& orig)
{
    int rows = curr.size();
    int cols = curr[0].size();
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
            curr[i][j] = curr[i][j] - orig[i][j];
}


///
/////
/////  MLP functions
/////
///


// use one of the init functions to initialize
mlp::mlp() { }


// Init model with labeled data, random weights
bool mlp::init_training_random(string fname, int _k, int _h)
{
    // sets X and X_labels
    // also sets n and d
    if(!read_data(X, X_labels, fname, true))
    {
        initialized = false;
        return initialized;
    }
    n = X.size();
    d = X[0].size();

    // Initialize weights
    // set k and h based on inputs
    k = _k;
    h = _h;

    // randomly fill W and V
    srand(0);  // seed for reproducability

    V.clear();
    V = vector<vector<double>>(h+1, vector<double>(k,0));
    for(int i = 0; i < h + 1; i++)
        for(int j = 0; j < k; j++)
            V[i][j] = -0.01 + (0.02 * ((double) rand() / RAND_MAX));

    W.clear();
    W = vector<vector<double>>(d+1, vector<double>(h,0));
    for(int i = 0; i < d + 1; i++)
        for(int j = 0; j < h; j++)
            W[i][j] = -0.01 + (0.02 * ((double) rand() / RAND_MAX));

    // Initialize Y and Z via one forwards propogation
    forward_propogate(X);

    initialized = true;
    return initialized;
}


// Init model with labeled data and input weights
// ASSUMES V and W are properly sized for the corresponding fname data
bool mlp::init_training_model(string fname, vector<vector<double>>& _V, vector<vector<double>>& _W)
{
    // sets X and X_labels
    // also sets n and d
    if(!read_data(X, X_labels, fname, true))
    {
        initialized = false;
        return initialized;
    }
    n = X.size();
    d = X[0].size();

    // sets weights of V and W
    // sets h and k
    set_weights(_V, _W);

    // Initialize Y and Z via one forwards propogation
    forward_propogate(X);

    initialized = true;
    return initialized;
}


// Train the model, needs to be initialized first
// Returns error after training or -1 on error
double mlp::train(double eta, int epochs)
{
    if(!initialized)
        return -1;

    // forwards propogate, initialize Y and Z
    forward_propogate(X);
    double err = error_func();

    // init dV dW to all 0s (initializes the size of dW and dV)
    vector<vector<double>> dV(h+1, vector<double>(k,0));
    vector<vector<double>> dW(d+1, vector<double>(h,0));
    
    // set dimensions of old dV and old dW
    vector<vector<double>> old_dV = dV;
    vector<vector<double>> old_dW = dW;

    for (int i = 0; i < epochs; i++)
    {
        // backwards propogate
        // this sets values in dV and dW
        backward_propogate(dV, dW, eta);

        // update with momentum
        // Add last iteration's dV / dW scaled by the momentum factor
        scale_matricies(old_dV, momentum);
        scale_matricies(old_dW, momentum);
        sum_matricies(dV, old_dV);
        sum_matricies(dW, old_dW);

        // Update old weights for momentum
        old_dV = dV;
        old_dW = dW;
        
        // update the weights from backwards propogation
        update_weights(dV, dW);

        // Update Z and Y with forwards propogation
        forward_propogate(X);

        // stop early if abs value difference between err and err_upd is <= 0.2
        double err_upd = error_func();
        if(abs(err - err_upd) <= 0.2 && i >= 25)
        {
            break;
        }
        err = err_upd;
    }

    // return final error rate
    return error_rate(X_labels);
}


// Run model on validation data
// Returns error after validation or -1 on error
double mlp::validate(string fname)
{
    if(!initialized)
        return -1;

    // read in validation data
    vector<int> _X_labels;
    vector<vector<int>> _X;
    if(!read_data(_X, _X_labels, fname, true))
        return -1;

    // cannot validate if no data
    // cannot validate if dimension of data =\=
    if(_X.size() < 1 || _X[0].size() != d)
        return -1;

    // do forward_propogation once to calculate Y and Z
    forward_propogate(_X);
    return error_rate(_X_labels);
}


// Make predicions on data
// ASSUMES data WITH NO LABELS has been read in
// Returns int vector of predictions - this vector is empty if error
vector<int> mlp::predict(string fname)
{
    vector<int> predictions;
    if(!initialized)
        return predictions;

    // read in data to make predictions on
    // (predictions is not set by read_data)
    vector<vector<int>> _X;
    if(!read_data(_X, predictions, fname, false))
        return predictions;

    // cannot validate if no data
    // cannot validate if dimension of data =\=
    if(_X.size() < 1 || _X[0].size() != d)
        return predictions;

    // do forward_propogation once to calculate Y and Z
    forward_propogate(_X); 
    for(int i = 0; i < n; i++)
    {
        int max_idx = 0;
        double max = Y[i][0];
        for(int j = 0; j < d; j++)
        {
            if(Y[i][j] > max)
            {
                max_idx = j;
                max = Y[i][j];
            }
        }
        predictions.push_back(max_idx);
    }
    return predictions;
}


///
/////  Weights functions
///


// Copies _W and _V to W and V
// also sets variables h and k
void mlp::set_weights(vector<vector<double>>& _V, vector<vector<double>>& _W)
{
    // set h and k based on input V and W
    h = _V.size() - 1;
    k = _V[0].size();
    V = _V;
    W = _W;
}

// Copies W and V to input _V and _W
void mlp::get_weights(vector<vector<double>>& _V, vector<vector<double>>& _W)
{
    _V = V;
    _W = W;
}

// update weights
// ASSUMES dV and dW are the same size as V and W respectively
void mlp::update_weights(vector<vector<double>>& _dV, vector<vector<double>>& _dW)
{
    for(int i = 0; i < h + 1; i++)
        for(int j = 0; j < k; j++)
            V[i][j] += _dV[i][j];
    
    for(int i = 0; i < d + 1; i++)
        for(int j = 0; j < h; j++)
            W[i][j] += _dW[i][j];
}


///
/////  Propogation functions
/////
/////  If you took CSCI 5521 last semester
/////  This may look familiar to you.  This
/////  is a translation of a MLP neural network
/////  from MATLAB to cpp (homework #5)
///


// Forwards propogation
// Uses datapoints in _X array to calculate Y and Z
void mlp::forward_propogate(vector<vector<int>>& _X)
{
    // Get number of datapoints in matrix _X
    int _n = _X.size();

    // Clear and size Z
    Z.clear();
    Z = vector<vector<double>>(_n, vector<double>(h+1,0));

    // Z = ReLU (X * W)
    // [n x d][d + 1 x h]
    for(int i = 0; i < _n; i++)
    {
        for (int j = 0; j < h; j++)
        {
            double dot = 0;
            for(int m = 0; m < d; m++)
                dot += _X[i][m] * W[m+1][j];
            dot += W[0][j];

            Z[i][j+1] = activation_func(dot);
        }
        Z[i][0] = 1;  // Z0 = 1
    }

    // Clear and size Y
    Y.clear();
    Y = vector<vector<double>>(_n, vector<double>(k,0));

    // O = Z * V
    // [n x h + 1][h + 1 x k]
    vector<vector<double>> O = vector<vector<double>>(_n, vector<double>(k, 0));
    for(int i = 0; i < _n; i++)
    {
        for(int j = 0; j < k; j++)
        {
            double dot = 0;
            for(int m = 0; m < h + 1; m++)
                dot += Z[i][m] * V[m][j];

            O[i][j] = dot;
        }
    }
    
    // Y(i,j) = 1 / sum(exp(O(i,:)-O(i,j)));
    for(int i = 0; i < _n; i++)
    {
        for(int j = 0; j < k; j++)
        {
            double sum = 0;
            for(int m = 0; m < k; m++)
                sum += exp(O[i][m] - O[i][j]);
            Y[i][j] = 1 / (sum + 0.000001);
        }
    }
}


// Backwards propogation
// Stores calcaulted weights updates in references _dV and _dW
void mlp::backward_propogate(vector<vector<double>>& _dV, vector<vector<double>>& _dW, double eta)
{
    // Initialize R [n x k] as rit - yit
    vector<vector<double>> R;
    R = vector<vector<double>>(n, vector<double>(k, 0));
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < k; j++)
        {
            if(X_labels[i] == j)
                R[i][j] = 1;

            R[i][j] -= Y[i][j];
        }
    }

    // dV update
    // (Rt * Z)t
    // [k x n][n x h + 1] -> [h + 1 x k]
    for(int i = 0; i < h + 1; i++)
    {
        for(int j = 0; j < k; j++)
        {
            double dot = 0;
            for(int m = 0; m < n; m++)
                dot += R[m][j] * Z[m][i];

            _dV[i][j] = eta * dot;
        }
    }

    // dW update
    // ((R * Vt(no bias) .* (X*W >= 0)t) * X)t
    // [n x k][k x h + 1]t  [n x d]  --->> [d x h + 1]

    // XW = (x * W) >= 0
    // [n x d][d + 1 x h]
    vector<vector<double>> XW;
    XW = vector<vector<double>>(n, vector<double>(h, 0));
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < h; j++)
        {
            double dot = 0;
            for(int m = 0; m < d; m++)
                dot += X[i][m] * W[m+1][j];
            dot += W[0][j];
            
            if(dot >= 0)
                XW[i][j] = 1;
        }
    }
    
    // S = (R * Vt(no bias) .* XW)'
    // [n x k][k x h]  -> [h x n]
    vector<vector<double>> S;
    S = vector<vector<double>>(h, vector<double>(n, 0));
    for(int i = 0; i < h; i++)
    {
        for(int j = 0; j < n; j++)
        {
            double dot = 0;
            for(int m = 0; m < k; m++)
                dot += R[j][m] * V[i+1][m];

            S[i][j] = dot * XW[j][i];
        }
    }

    // dW = (S * X)'
    // [h x n][n x d] -> [d + 1 x h]
    for(int i = 0; i < d; i++)
    {
        for(int j = 0; j < h; j++)
        {
            double dot = 0;
            for(int m = 0; m < n; m++)
                dot += S[j][m] * X[m][i];

            _dW[i+1][j] = eta * dot;
        }
    }
    
    // update dW[0][:]
    for(int i = 0; i < h; i++)
    {
        double sum = 0;
        for(int j = 0; j < n; j++)
            sum += S[i][j];
        _dW[0][i] = eta * sum;
    }   
}


///
/////  Other Private Functions
///


// Read CSV file of data
// ASSUMES all input datapoints are integers
// ASSUMES labels come at the end of the line if included
bool mlp::read_data(vector<vector<int>>& _X, vector<int>& _X_labels, string fname, bool labels)
{
    fstream data;
    data.open(fname, ios::in);
    if(!data.is_open())
        return false;

    // clear any pre-existing data
    _X.clear();
    _X_labels.clear();

    // parse lines
    string line;
    while(getline(data, line))
    {   
        string sdata = "";
        vector<int> datapoint;
        for(char c : line)
        {
            if(c == ',')
            {
                datapoint.push_back(stoi(sdata));
                sdata = "";
            }
            else
            {
                sdata += c;
            }
        }

        // Add last element to datapoint
        // If training data, add last element (label) to X_label
        if(!labels)
        {
            datapoint.push_back(stoi(sdata));
        }
        else
        {
            _X_labels.push_back(stoi(sdata));
        }

        // Add datapoint to data matrix X
        _X.push_back(datapoint);
    }
    data.close();
    return true;
}


// Calculate error of the model
// Tests Predicitons in Y against labels
// ASSUMES the model has been built (i.e Y is filled out)
double mlp::error_rate(vector<int>& _X_labels)
{
    double total = 0;
    double err_tot = 0;
    int _n = _X_labels.size();
    for(int i = 0; i < _n; i++)
    {
        int max_idx = 0;
        double max = -9999;
        for(int j = 0; j < k; j++)
        {
            if(Y[i][j] > max)
            {
                max = Y[i][j];
                max_idx = j;
            }
        }
        if(_X_labels[i] != max_idx)
            err_tot++;
        total++;
    }
    return err_tot / total;
}


// Calculate error of the model
double mlp::error_func()
{
    double err = 0;
    for(int i = 0; i < n; i++)
    {
        err += log(Y[i][X_labels[i]] + 0.000001);
    }
    return -err;
}


// Activation function, Using ReLU
double mlp::activation_func(double x)
{
    if(0 > x)
        return 0;
    return x;
}
