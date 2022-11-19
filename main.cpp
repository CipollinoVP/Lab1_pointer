#include <iostream>
#include "omp.h"
#include <ctime>
#include <cmath>
#include <fstream>


using namespace std;

//зануляет матрицу
void clear_matrix(double* A, const int N) {
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k)
            A[i * N + k] = 0;
}

//обычное умножение матриц для проверки ||A_ishod - L*U||
void prod(const double* A, const double* B, double*& C, const int N) {
    clear_matrix(C,N);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}


// копирует матрицу B в матрицу C
void copy_matrix(const double* B, double*& C, const int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = B[i * N + j];
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


void rand_fill(double* A, const int N) {
    srand(time(nullptr));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (rand() % 100 + 1);
        }
}

// обычное LU (для квадратных)
void LU(double* A, const int n, int m) {
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

// обычное LU (для квадратных) параллельное
void LU_parallel(double* A, const int n, const int m)
{
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

void inverse_L(double* & L, double* & L1, int n, int n1){
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            L1[i*n+j] = 0;
        }
        L1[i*n+i] = 1;
    }
    for (int i = 0; i < n-1; ++i) {
        for (int k = i+1; k < n; ++k) {
            for (int j = 0; j < i+1; ++j) {
                L1[k*n+j] -= L1[i*n+j]*L[k*n1+i];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            L[i*n1+j] = L1[i*n+j];
        }
    }
}


void inverse_L_parallel(double* & L, double* & L1, int n, int n1){
    for (int i = 0; i < n; ++i) {
#pragma omp parallel for default(none) shared(L,n,L1,n1,i)
        for (int j = 0; j < n; ++j) {
            L1[i*n+j] = 0;
        }
        L1[i*n+i] = 1;
    }
    for (int i = 0; i < n-1; ++i) {
        for (int k = i+1; k < n; ++k) {
#pragma omp parallel for default(none) shared(L,n,i,L1,k,n1)
            for (int j = 0; j < i+1; ++j) {
                L1[k*n+j] -= L1[i*n+j]*L[k*n1+i];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
#pragma omp parallel for default(none) shared(L,n,i,L1,n1)
        for (int j = 0; j < i; ++j) {
            L[i*n1+j] = L1[i*n+j];
        }
    }
}

//алгоритм 2.10
void two_ten(double*& A, int n, int b) {
    int a1 = n;
    int a2 = n - b;
    auto* L2232 = new double[a1 * b];
    auto* U23 = new double[b * a2];
    auto* Lh = new double[n*n];
    for (int i = 0; i < n - 1; i += b) {
        a1 = (n - i);
        a2 = (n - b - i);

        //(1)
        for (int y = 0; y < a1; y++)
            for (int z = 0; z < b; z++) {
                L2232[y * b + z] = A[(y + i) * n + (i + z)];
            }

        LU(L2232, a1, b);

        for (int y = 0; y < (a1); y++)
            for (int z = 0; z < b; z++) {
                A[(y + i) * n + (i + z)] = L2232[y * b + z];
            }
        //(1)

        if ((n - b - i) > 0) {
            //(2)
            for (int y = 0; y < b; y++) {
                for (int z = 0; z < a2; z++) {
                    U23[y*a2 + z] = A[(y + i) * n + z + i + b];
                }
            }
            inverse_L(L2232,Lh,b,b);
            for (int y = 0; y < b; y++) {
                for (int z = 0; z < (a2); z++) {
                    A[(y + i) * n + z + i + b] = 0;
                    for (int j = 0; j < b; ++j) {
                        if (y>j) {
                            A[(y + i) * n + z + i + b] += L2232[y * b + j] * U23[j * a2 + z];
                        } else if (y==j){
                            A[(y + i) * n + z + i + b] += U23[j * a2 + z];
                        }
                    }
                }
            }
            //(2)

            //(3)
            for (int y = 0; y < b; y++) {
                for (int z = 0; z < a2; z++) {
                    U23[y*a2 + z] = A[(y + i) * n + z + i + b];
                }
            }
            for (int p = b; p < a1; ++p) {
                for (int k = b; k < a1; ++k) {
                    for (int q = 0; q < b; ++q) {
                        A[(i+p)*n+ i+ k] -= L2232[p * b + q] * U23[q * a2 + k - b];        //(n-i-b) это a2
                    }
                }
            }
            //(3)
        }
    }
    delete[] L2232;
    delete[] U23;
    delete[] Lh;
}




//Алгоритм 2.10 параллельный
void two_ten_parallel(double*& A, int n, int b) {
    int a1 = n;
    int a2 = n - b;
    auto* L2232 = new double[a1 * b];
    auto* U23 = new double[b * a2];
    auto* Lh = new double[b*b*2];
    for (int i = 0; i < n - 1; i += b) {
        a1 = (n - i);
        a2 = (n - b - i);

        //(1)
        for (int y = 0; y < a1; y++)
            for (int z = 0; z < b; z++) {
                L2232[y * b + z] = A[(y + i) * n + (i + z)];
            }

        LU_parallel(L2232, a1, b);

        for (int y = 0; y < (a1); y++)
            for (int z = 0; z < b; z++) {
                A[(y + i) * n + (i + z)] = L2232[y * b + z];
            }
        //(1)

        if ((n - b - i) > 0) {
            //(2)
            for (int y = 0; y < b; y++) {
                for (int z = 0; z < a2; z++) {
                    U23[y*a2 + z] = A[(y + i) * n + z + i + b];
                }
            }
            inverse_L_parallel(L2232,Lh,b,b);
#pragma omp parallel for default(none) shared(L2232,n,i,A,b,a2,U23)
            for (int y = 0; y < b; y++) {
                for (int z = 0; z < (a2); z++) {
                    A[(y + i) * n + z + i + b] = 0;
                    for (int j = 0; j < b; ++j) {
                        if (y>j) {
                            A[(y + i) * n + z + i + b] += L2232[y * b + j] * U23[j * a2 + z];
                        } else if (y==j){
                            A[(y + i) * n + z + i + b] += U23[j * a2 + z];
                        }
                    }
                }
            }
            //(2)

            //(3)
            for (int y = 0; y < b; y++) {
                for (int z = 0; z < a2; z++) {
                    U23[y*a2 + z] = A[(y + i) * n + z + i + b];
                }
            }
#pragma omp parallel for default(none) shared(L2232,n,i,A,b,U23,a2,a1)
            for (int p = b; p < a1; ++p) {
                for (int k = b; k < a1; ++k) {
                    for (int q = 0; q < b; ++q) {
                        A[(i+p)*n+ i+ k] -= L2232[p * b + q] * U23[q * a2 + k - b];        //(n-i-b) это a2
                    }
                }
            }
            //(3)
        }
    }
    delete[] L2232;
    delete[] U23;
    delete[] Lh;
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


int main() {
    std::ofstream out("/nethome/student/FS19/FS2-x1/Popov_Kiseleva/Lab1_pointer/out.txt");
    omp_set_dynamic(0);
    omp_set_num_threads(8);
    int a = 5;
    srand(time(0));
    int n = 512;
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
        double t1,t2;
        t1 = omp_get_wtime();
        LU(B1,n,m);
        t2 = omp_get_wtime();;
        double time1 = t2-t1;
        t1 = omp_get_wtime();
        LU_parallel(B2,n,m);
        t2 = omp_get_wtime();
        double err1 = 0;
        for (int i = 0; i < n*m; ++i) {
            if (err1 < std::abs(B1[i]-B2[i])) {
                err1 = std::abs(B1[i]-B2[i]);
            }
        }
        double time2 = t2-t1;
        int block = 512;
        t1 = omp_get_wtime();
        two_ten(B3,n,block);
        t2 = omp_get_wtime();
        double err2 = 0;
        for (int i = 0; i < n*m; ++i) {
            if (err2 < std::abs(B1[i]-B3[i])) {
                err2 = std::abs(B1[i]-B3[i]);
            }
        }
        double time3 = t2-t1;
        t1 = omp_get_wtime();
        two_ten_parallel(B4,n,block);
        t2 = omp_get_wtime();
        double err3 = 0;
        for (int i = 0; i < n*m; ++i) {
            if (err3 < std::abs(B1[i]-B4[i])) {
                err3 = std::abs(B1[i]-B4[i]);
            }
        }
        double time4 = t2-t1;
        out << n << std::endl << "Неблочное LU-разложение без распараллеливания" << std::endl << "Время: " <<
                  time1 << std::endl <<"Неблочное LU-разложение с распараллеливанием" << std::endl << "Время " << time2 <<
                  "  Ошибка в сравнении с первыи разложением: " << err1 << std::endl
                  << "Ускорение " << time1/time2 << std::endl
                  << "Блочное LU-разложение без распараллеливания"<< std::endl << "Время: " << time3 << "  Ошибка в сравнении с первыи разложением: "
                  << err2 << std::endl
                  << "Блочное LU-разложение с распараллеливанием" << std::endl << "Время: " << time4
                  << "  Ошибка в сравнении с первыи разложением: " << err3 << std::endl << "Ускорение " << time3/time4 <<
                  std::endl << "Блочный/не блочный 1" << std::endl << time2/time1 << std::endl << "Блочный/не блочный 2" <<
                  std::endl << time4/time3 << std::endl << "Максимальное отношение времени:" << std::endl <<
                  time4/time1 << std::endl;
        delete[] A;
        delete[] B1;
        delete[] B2;
        delete[] B3;
        delete[] B4;
    std::cout << "Готово";
    return 0;
}
