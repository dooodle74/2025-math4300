def reciprocal(a, x_n, tolerance, max_iter, iter_count = 0):

    if iter_count >= max_iter:
        return x_n 

    x_next = x_n * (2 - a * x_n)

    if abs(x_next - x_n) < tolerance: 
        return x_next
    
    return reciprocal(a, x_next, tolerance, max_iter, iter_count + 1)

def main():
    a = 37
    x_0 = 0.01
    max_iter = 10
    tolerance = 5e-4

    print(reciprocal(a, x_0, tolerance, max_iter))

if __name__=="__main__":
    main()