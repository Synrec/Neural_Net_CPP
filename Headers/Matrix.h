#include <vector>
#include <cmath>
#include <cassert>
#include <functional>

class Matrix
{
    public:
    unsigned _cols;
    unsigned _rows;
    std::vector<float> _vals;
        Matrix()
            :_cols(0), _rows(0), _vals({})
        {}

        Matrix(unsigned cols, unsigned rows)
            :_cols(cols), _rows(rows), _vals({})
        {
            _vals.resize(cols * rows, 0.0f);
        }

        Matrix applyFunction(std::function<float(const float&)> func)
        {
            Matrix output(_cols, _rows);
            for(unsigned y = 0; y < output._rows; y++){
                for(unsigned x = 0; x < output._cols; x++){
                    output.at(x, y) = func(at(x, y));
                }
            }
            return output;
        }

        float &at(unsigned col, unsigned row){
            return _vals[row * _cols + col];
        }

        Matrix multiply(Matrix target)
        {
            assert(_cols == target._rows);
            Matrix output(target._cols, _rows);
            for(unsigned y = 0; y < output._rows; y++){
                for(unsigned x = 0; x < output._cols; x++){
                    float result = 0.0f;
                    for(unsigned k = 0; k < _cols; k++){
                        result += at(k, y) * target.at(x, k);
                    }
                    output.at(x, y) = result;
                }
            }
            return output;
        }

        Matrix multiply_scalar(float s)
        {
            Matrix output(_cols, _rows);
            for(unsigned y = 0; y < output._rows; y++){
                for(unsigned x = 0; x < output._cols; x++){
                    output.at(x, y) = at(x, y) * s;
                }
            }
            return output;
        }

        Matrix multiply_elements(Matrix &target)
        {
            assert(target._rows == _rows && target._cols == _cols);
            Matrix output(_cols, _rows);
            for(unsigned y = 0; y < output._rows; y++){
                for(unsigned x = 0; x < output._cols; x++){
                    output.at(x, y) = at(x, y) * target.at(x, y);
                }
            }
            return output;
        }

        Matrix add(Matrix target)
        {
            assert(_rows == target._rows && _cols == target._cols);
            Matrix output(_cols, _rows);
            for(unsigned y = 0; y < output._rows; y++){
                for(unsigned x = 0; x < output._cols; x++){
                    output.at(x, y) = at(x,y) + target.at(x,y);
                }
            }
            return output;
        }

        Matrix add_scalar(float s)
        {
            Matrix output(_cols, _rows);
            for(unsigned y = 0; y < output._rows; y++){
                for(unsigned x = 0; x < output._cols; x++){
                    output.at(x, y) = at(x, y) + s;
                }
            }
            return output;
        }

        Matrix negative(void)
        {
            Matrix output(_cols, _rows);
            for(unsigned y = 0; y < output._rows; y++){
                for(unsigned x = 0; x < output._cols; x++){
                    output.at(x, y) = -at(x, y);
                }
            }
            return output;
        }

        Matrix transpose()
        {
            Matrix output(_rows, _cols);
            for(unsigned y = 0; y < _rows; y++){
                for(unsigned x = 0; x < _cols; x++){
                    output.at(y, x) = at(x, y);
                }
            }
            return output;
        }
};