#include<cstdio>
#include<cassert>

float get_topk(float* arr, int len, int k){
    assert
    // cout << "arr0 " << arr << endl;
    Matrix<float, 1, Dynamic> l_arr = Matrix<float, 1, Dynamic>::Constant(1, length, -10000);
    Matrix<float, 1, Dynamic> r_arr = Matrix<float, 1, Dynamic>::Constant(1, length, -10000);
    int iter = 0;
    while (true){
        // cout << " " << endl;
        if (iter > 2000) return -1;
        // cout << "arr " << arr << endl;
        float pivot = arr(0,0);
        int length = arr.cols();
        if (iter % 2 == 0) pivot = arr(0, length - 1);
        // cout << "pivot " << pivot << endl;
        int l_counter = 0;
        int r_counter = 0;
        for (int i = 0; i < length; i++){
            if (arr(0,i) <= pivot) l_arr(0,l_counter++) = arr(0,i);
            else r_arr(0, r_counter++) = arr(0, i);
        }
        // cout << "l_counter " << l_counter << endl;
        // cout << "r_counter " << r_counter << endl;
        if (r_counter == k - 1) return pivot;
        else if (r_counter < k-1) {
            arr = l_arr.leftCols(l_counter);
            k = k - r_counter;
        }
        else{
            arr = r_arr.leftCols(r_counter);
            // k = k - l_counter;
        }
        if (arr.cols() == 1) return arr(0,0);
        iter++;
        // cout << "k " << k << endl;
    }

}