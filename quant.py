import numpy as np
import matplotlib.pyplot as plt


def find_quantization_points(pdf, bounds, num_points):
    # Generate x values
    x = np.linspace(bounds[0], bounds[1], 10000)
    # Calculate the PDF values
    y = pdf(x)
    # Calculate the area under the PDF (using the trapezoidal rule)
    dx = x[1] - x[0]
    cumulative_area = np.cumsum(y) * dx
    total_area = cumulative_area[-1]

    # Determine target areas for each quantile
    target_areas = np.linspace(0, total_area, num_points + 1)[1:-1]

    # Find indices where the cumulative area surpasses each target area
    quantization_indices = np.searchsorted(cumulative_area, target_areas)
    quantization_points = x[quantization_indices]

    return quantization_points


# Parameters for the normal distribution
mu = 0
sigma = 1


def normal_pdf(x):
    return (1/(sigma * np.sqrt(np.pi)) *
            np.exp(-1/2 * np.power((x - mu)/sigma, 2)))


# Quantization
num_quant_points = 100  # Number of segments (not points)
bounds = [-3, 3]  # Effective range for normal distribution around the mean
quantization_points = find_quantization_points(
    normal_pdf, bounds, num_quant_points)

# Plotting
x_range = np.linspace(bounds[0], bounds[1], 1000)
plt.figure(figsize=(8, 5))
plt.plot(x_range, normal_pdf(x_range), label="Normal Distribution PDF")

plt.scatter(quantization_points, normal_pdf(
    np.array(quantization_points)), color='red', s=2)

plt.plot(quantization_points, normal_pdf(
    np.array(quantization_points)), color='red')

plt.plot(np.gradient(quantization_points))

# for point in quantization_points:
#     plt.annotate(f'{point:.2f}', (point, normal_pdf(point)),
#                  textcoords="offset points", xytext=(0, 10), ha='center')
plt.title('Normal PDF and Quantization Points')
plt.xlabel('x')
plt.ylabel('PDF(x)')
plt.legend()
plt.grid(True)
plt.show()

# Output quantization points and their corresponding PDF values
# for point in quantization_points:
#     print(f'x: {point:.4f}, PDF(x): {normal_pdf(point):.4f}')
