//
// Created by Marco DiVi on 02/04/18.
//

#ifndef PROJECTWORK_INFINITEMATRIX_H
#define PROJECTWORK_INFINITEMATRIX_H

#include "opencv2/core.hpp"
#include <type_traits>

//todo: also handle multiple channels
//todo: use instead of normal matrix to handle votes and scales in ght (so that you don't need to handle outliers differently)


template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct InfiniteMatrix {

    private:
        cv::Mat _mat;
        int rows, cols;
        int zero_x = 0, zero_y = 0;

    public:

        explicit InfiniteMatrix(int rows, int cols) {

            this->rows = rows;
            this->cols = cols;

            if (std::is_same<T, int>::value)
                _mat = cv::Mat::zeros(rows, cols, CV_32SC1);
            else
                _mat = cv::Mat::zeros(rows, cols, CV_32FC1);

        }

        cv::Mat asMat(bool cropped=false) {
            if (!cropped)
                return _mat;

            cv::Mat newmat = cv::Mat::zeros(rows, cols, _mat.type());

            for (int i = zero_x; i < zero_x + rows; i++) {
                for (int j = zero_y; j < zero_y + cols; j++) {
                    newmat.at<T>(i, j) = _mat.at<T>(i - zero_x, j - zero_y);
                }
            }

            return newmat;
        }

        void set(int row, int col, T value) {

            if (row >= 0 && row < _mat.rows && col >= 0 && col < _mat.cols){
                _mat.at<T>(row, col) = value;
                return;
            }

            int newrows = 0, newcols = 0;

            if (row < 0) {
                newrows = -row;
                zero_x += row;
            }
            else if (row >= _mat.rows) newrows = row - _mat.rows + 1;

            if (col < 0) {
                newcols = -col;
                zero_y += col;
            }
            else if (col >= _mat.cols) newcols = col - _mat.cols + 1;

            cv::Mat newmat = cv::Mat::zeros(_mat.rows + newrows, _mat.cols + newcols, _mat.type());

            for (int i = zero_x; i < zero_x + rows; i++) {
                for (int j = zero_y; j < zero_y + cols; j++) {
                    newmat.at<T>(i, j) = _mat.at<T>(i - zero_x, j - zero_y);
                }
            }

            _mat = newmat;

        }

        /// use infiniteCoord to access negative coords within the image, (0,0) will be the original (0,0)
        // with this, you don't need to get the cropped image to access "normal" points
        T get(int row, int col, bool infiniteCoord = false) {

            if (infiniteCoord) {
                row += zero_x;
                col += zero_y;
            }

            return _mat.at<T>(row, col);
        }

};


#endif //PROJECTWORK_INFINITEMATRIX_H
