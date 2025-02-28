import math

def linear_correlation_coefficient(x, y):
    n = len(x)

    sum_x, sum_y, sum_x_sq, sum_y_sq, sum_xy = 0, 0, 0, 0, 0
    
    for i in range(n):
        sum_x += x[i]
        sum_x_sq += x[i] ** 2

        sum_y += y[i]
        sum_y_sq += y[i] ** 2

        sum_xy += x[i] * y[i]

    num = n * sum_xy - (sum_x * sum_y)
    denom = math.sqrt((n * sum_x_sq - sum_x**2) * (n * sum_y_sq - sum_y**2))

    if denom == 0:
        return float('nan')
    return num / denom

def main():
    x_values = [3, 7, 9, 2, 7, 0, 3]
    y_values = [-5, 10, 15, -8, 11, -10, -4]

    print(linear_correlation_coefficient(x_values, y_values))

if __name__=="__main__":
    main()