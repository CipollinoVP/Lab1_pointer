#include <iostream>
#include <cmath>
#include "omp.h"
#include "vector"
#include "ctime"
#include "fstream"

double* inverse_L(double* const& L, int n){
    auto* res = new double[n*n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i*n+j] = 0;
        }
        res[i*n+i] = 1;
    }
    for (int i = 0; i < n-1; ++i) {
        for (int j = i+1; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                res[j*n+k] = res[j*n+k] - res[i*n+k]*L[j*n+i];
            }
        }
    }
    return res;
}

double* inverse_L_parallel(double* const& L, int n){
    auto* res = new double[n*n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            res[i*n+j] = 0;
        }
        res[i*n+i] = 1;
    }
    for (int i = 0; i < n-1; ++i) {
#pragma omp parallel for default(none) shared(res,i,L,n)
        for (int j = i+1; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                res[j*n+k] = res[j*n+k] - res[i*n+k]*L[j*n+i];
            }
        }
    }
    return res;
}

double* prod(double* const& left, int l_n, int l_m,
             double* const& right, int r_n, int r_m){
    auto* res = new double[l_n*r_m];
    for (int i = 0; i < l_n; ++i) {
        for (int j = 0; j < r_m; ++j) {
            res[i*r_m + j] = 0;
            for (int k = 0; k < r_n; ++k) {
                res[i*r_m+j] = res[i*r_m+j] + left[i*l_m+k]*right[k*r_m+j];
            }
        }
    }
    return res;
}

double* prod_parallel(double* const& left, int l_n, int l_m,
             double* const& right, int r_n, int r_m){
    auto* res = new double[l_n*r_m];
#pragma omp parallel for default(none) shared(res,left,right,l_n,l_m,r_n,r_m)
    for (int i = 0; i < l_n; ++i) {
        for (int j = 0; j < r_m; ++j) {
            res[i*r_m + j] = 0;
            for (int k = 0; k < r_n; ++k) {
                res[i*r_m+j] = res[i*r_m+j] + left[i*l_m+k]*right[k*r_m+j];
            }
        }
    }
    return res;
}


void LU_parallel(double* &A, int n, int m){
    if (A == nullptr){
        return;
    }
    for (int i = 0; i < std::min(n-1,m); ++i) {
#pragma omp parallel for default(none) shared(A,i,n,m)
        for (int j = i+1; j < n; ++j) {
            A[j*m+i] = A[j*m+i]/A[i*m+i];
        }
        if (i<m){
#pragma omp parallel for default(none) shared(A,i,n,m)
            for (int j = i+1; j < n; ++j) {
                for (int k = i+1; k < m; ++k) {
                    A[j*m+k]=A[j*m+k]-A[j*m+i]*A[i*m+k];
                }
            }
        }
    }
}

void LU(double* &A, int n, int m){
    if (A == nullptr){
        return;
    }
    for (int i = 0; i < std::min(n-1,m); ++i) {
        for (int j = i+1; j < n; ++j) {
            A[j*m+i] = A[j*m+i]/A[i*m+i];
        }
        if (i<m){
            for (int j = i+1; j < n; ++j) {
                for (int k = i+1; k < m; ++k) {
                    A[j*m+k]=A[j*m+k]-A[j*m+i]*A[i*m+k];
                }
            }
        }
    }
}

void matrix_out(double* const& A, int n, int m){
    if (A == nullptr){
        return;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            std::cout << A[i*m+j] << "  ";
        }
        std::cout << std::endl;
    }
}

double* difference(double* const& right, int r_n, int r_m,
                   double* const& left){
    auto* result = new double[r_n*r_m];
    for (int i = 0; i < r_n; ++i) {
        for (int j = 0; j < r_m; ++j) {
            result[i*r_m+j] = right[i*r_m+j]-left[i*r_m+j];
        }
    }
    return result;
}

void LU_Blocks(double* &A, int n, int m, int b){
    for (int i = 0; i<n-1; i+=b)
    {
        auto* subA = new double[(m-i)*b];
        for (int j = 0; j < m-i; ++j) {
            for (int k = 0; k < b; ++k) {
                subA[j*b+k] = A[(j+i)*m+k+i];
            }
        }
        LU(subA,m-i,b);
        for (int j = 0; j < m-i; ++j) {
            for (int k = 0; k < b; ++k) {
                A[(j+i)*m+k+i] = subA[j*b+k];
            }
        }
        delete[] subA;
        if ((int) m-i - b > 0) {
            auto* subL = new double[b*b];
            for (int j = 0; j < b; ++j) {
                subL[j*b+j] = 1;
                for (int k = j; k < b; ++k) {
                    subL[j*b+k] = 0;
                }
                for (int k = 0; k < j; ++k) {
                    subL[j*b+k] = A[(j+i)*m+k+i];
                }
            }
            double* subL1 = inverse_L(subL,b);
            delete[] subL;
            subA = new double[b*(m-i - b)];
            for (int j = 0; j < b; ++j) {
                for (int k = b; k < m-i; ++k) {
                    subA[j*(m-i - b)+k - b] = A[(j+i)*m+k+i];
                }
            }
            double* subA1 = prod(subL1,b,b,subA,b,m-i-b);
            delete[] subL1;
            delete[] subA;
            for (int j = 0; j < b; ++j) {
                for (int k = b; k < m-i; ++k) {
                    A[(j+i)*m+k+i] = subA1[j*(m-i-b)+k - b];
                }
            }
            delete[] subA1;
            subA1 = new double[(m-i - b)*b];
            for (int j = b; j < m-i; ++j) {
                for (int k = 0; k < b; ++k) {
                    subA1[(j - b)*b+k] = A[(j+i)*m+k+i];
                }
            }
            auto* subA2 = new double[b*(m-i - b)];
            for (int j = 0; j < b; ++j) {
                for (int k = b; k < m-i; ++k) {
                    subA2[j*(m-i-b)+k - b] = A[(j+i)*m+k+i];
                }
            }
            subA = prod(subA1,m-i-b,b,subA2,b,m-i-b);
            delete[] subA1;
            delete[] subA2;
            for (int j = b; j < m-i; ++j) {
                for (int k = b; k < m-i; ++k) {
                    A[(j+i)*m+k+i] = A[(j+i)*m+i+k] - subA[(j - b)*(m-i-b)+k - b];
                }
            }
        }
    }
}

void LU_Blocks_parallel(double* &A, int n, int m, int b){
    for (int i = 0; i<n-1; i+=b)
    {
        auto* subA = new double[(m-i)*b];
#pragma omp parallel for default(none) shared(A,subA,i,m,b)
        for (int j = 0; j < m-i; ++j) {
            for (int k = 0; k < b; ++k) {
                subA[j*b+k] = A[(j+i)*m+k+i];
            }
        }
        LU_parallel(subA,m-i,b);
#pragma omp parallel for default(none) shared(A,subA,i,m,b)
        for (int j = 0; j < m-i; ++j) {
            for (int k = 0; k < b; ++k) {
                A[(j+i)*m+k+i] = subA[j*b+k];
            }
        }
        delete[] subA;
        if ((int) m-i - b > 0) {
            auto* subL = new double[b*b];
#pragma omp parallel for default(none) shared(A,subL,i,m,b)
            for (int j = 0; j < b; ++j) {
                subL[j*b+j] = 1;
                for (int k = j; k < b; ++k) {
                    subL[j*b+k] = 0;
                }
                for (int k = 0; k < j; ++k) {
                    subL[j*b+k] = A[(j+i)*m+k+i];
                }
            }
            double* subL1 = inverse_L_parallel(subL,b);
            delete[] subL;
            subA = new double[b*(m-i - b)];
#pragma omp parallel for default(none) shared(A,subA,i,m,b)
            for (int j = 0; j < b; ++j) {
                for (int k = b; k < m-i; ++k) {
                    subA[j*(m-i - b)+k - b] = A[(j+i)*m+k+i];
                }
            }
            double* subA1 = prod_parallel(subL1,b,b,subA,b,m-i-b);
            delete[] subL1;
            delete[] subA;
#pragma omp parallel for default(none) shared(A,subA1,i,m,b)
            for (int j = 0; j < b; ++j) {
                for (int k = b; k < m-i; ++k) {
                    A[(j+i)*m+k+i] = subA1[j*(m-i-b)+k - b];
                }
            }
            delete[] subA1;
            subA1 = new double[(m-i - b)*b];
#pragma omp parallel for default(none) shared(A,subA1,i,m,b)
            for (int j = b; j < m-i; ++j) {
                for (int k = 0; k < b; ++k) {
                    subA1[(j - b)*b+k] = A[(j+i)*m+k+i];
                }
            }
            auto* subA2 = new double[b*(m-i - b)];
#pragma omp parallel for default(none) shared(A,subA2,i,m,b)
            for (int j = 0; j < b; ++j) {
                for (int k = b; k < m-i; ++k) {
                    subA2[j*(m-i-b)+k - b] = A[(j+i)*m+k+i];
                }
            }
            subA = prod_parallel(subA1,m-i-b,b,subA2,b,m-i-b);
            delete[] subA1;
            delete[] subA2;
#pragma omp parallel for default(none) shared(A,subA,i,m,b)
            for (int j = b; j < m-i; ++j) {
                for (int k = b; k < m-i; ++k) {
                    A[(j+i)*m+k+i] = A[(j+i)*m+i+k] - subA[(j - b)*(m-i-b)+k - b];
                }
            }
        }
    }
}

int main() {
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    int a = 5;
    srand(time(0));
    int n = 4096;
    int m;
    m = n;
    auto* A = new double[n*m];
    auto* B1 = new double[n*m];
    auto* B2 = new double[n*m];
    auto* B3 = new double[n*m];
    auto* B4 = new double[n*m];
    for (int i = 0; i < n*m; ++i) {
        A[i] = (double)rand() /RAND_MAX;
        B1[i] = A[i];
        B2[i] = A[i];
        B3[i] = A[i];
        B4[i] = A[i];
    }
    long int t1 = clock();
    LU(B1,n,m);
    long int t2 = clock();
    double time1 = (double) (t2-t1)/CLOCKS_PER_SEC;
    t1 = clock();
    LU_parallel(B2,n,m);
    t2 = clock();
    double err1 = 0;
    for (int i = 0; i < n*m; ++i) {
        if (err1 < std::abs(B1[i]-B2[i])) {
            err1 = std::abs(B1[i]-B2[i]);
        }
    }
    double time2 = (double) (t2-t1)/CLOCKS_PER_SEC;
    t1 = clock();
    int block = 64;
    LU_Blocks(B3,n,m,block);
    t2 = clock();
    double err2 = 0;
    for (int i = 0; i < n*m; ++i) {
        if (err2 < std::abs(B1[i]-B3[i])) {
            err2 = std::abs(B1[i]-B3[i]);
        }
    }
    double time3 = (double) (t2-t1)/CLOCKS_PER_SEC;
    t1 = clock();
    LU_Blocks_parallel(B4,n,m,block);
    t2 = clock();
    double err3 = 0;
    for (int i = 0; i < n*m; ++i) {
        if (err3 < std::abs(B1[i]-B4[i])) {
            err3 = std::abs(B1[i]-B4[i]);
        }
    }
    double time4 = (double) (t2-t1)/CLOCKS_PER_SEC;
    std::cout << "Неблочное LU-разложение без распараллеливания" << std::endl << "Время: " <<
              time1 << std::endl <<"Неблочное LU-разложение с распараллеливанием" << std::endl << "Время " << time2 <<
              "  Ошибка в сравнении с первыи разложением: " << err1 << std::endl
              << "Ускорение " << time1/time2 << std::endl
              << "Блочное LU-разложение без распараллеливания"<< std::endl << "Время: " << time3 << "  Ошибка в сравнении с первыи разложением: "
              << err2 << std::endl
              << "Блочное LU-разложение с распараллеливанием" << std::endl << "Время: " << time4
              << "  Ошибка в сравнении с первыи разложением: " << err3 << std::endl << "Ускорение " << time3/time4 << std::endl;
    return 0;
}
