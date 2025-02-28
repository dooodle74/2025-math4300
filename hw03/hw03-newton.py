import numpy as np
import pandas as pd

def newtons_method(f, df, p0, max_iter=4):
    """Performs Newton's method and stores results in a table."""
    results = []
    
    for i in range(max_iter):
        f_p0 = f(p0)
        df_p0 = df(p0)
        p_next = p0 - f_p0 / df_p0
        results.append((i, p0, f_p0, df_p0, p_next))
        p0 = p_next  # Update for next iteration

    df_results = pd.DataFrame(results, columns=['n', 'p_n', 'f(p_n)', "f'(p_n)", 'p_{n+1}'])
    return df_results

def to_latex_table(df, function_name):
    """Converts a DataFrame to LaTeX table source code."""
    latex_code = df.to_latex(index=False, float_format="%.10f", escape=False)
    latex_code = latex_code.replace("\\toprule", "\\hline").replace("\\midrule", "\\hline").replace("\\bottomrule", "\\hline")
    table = f"\\begin{{table}}[h]\n\\centering\n\\caption{{Newton's Method for {function_name}}}\n{latex_code}\n\\end{{table}}"
    return table

# Define functions and their manually computed derivatives
def f1(x):  # ln(1 + x) - cos(x)
    return np.log(1 + x) - np.cos(x)

def df1(x):  # Derivative: 1/(1 + x) + sin(x)
    return 1 / (1 + x) + np.sin(x)

def f2(x):  # x^5 + 2x - 1
    return x**5 + 2*x - 1

def df2(x):  # Derivative: 5x^4 + 2
    return 5 * x**4 + 2

def f3(x):  # e^(-x) - x
    return np.exp(-x) - x

def df3(x):  # Derivative: -e^(-x) - 1
    return -np.exp(-x) - 1

def f4(x):  # cos(x) - x
    return np.cos(x) - x

def df4(x):  # Derivative: -sin(x) - 1
    return -np.sin(x) - 1

# Initial guesses for each function
initial_guesses = {
    "ln(1 + x) - cos(x)": (f1, df1, 0.5),
    "x^5 + 2x - 1": (f2, df2, 0.5),
    "e^(-x) - x": (f3, df3, 0.5),
    "cos(x) - x": (f4, df4, 0.5),
}

# Perform Newton's method and generate LaTeX tables
for name, (f, df, p0) in initial_guesses.items():
    df_results = newtons_method(f, df, p0, max_iter=4)
    latex_code = to_latex_table(df_results, name)
    print(f"\nLaTeX Table for {name}:\n")
    print(latex_code)