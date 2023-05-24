import numpy as np
import matplotlib.pyplot as plt
import fitcode

# Array of (Voltage, Lower radius, Upper radius, Amperage)
data1 = np.array([[80.2, 290.0, 351.4, 1.001], [99.9, 285.4, 366.1, 1.000], [119.6, 280.0, 370.1, 1.001],
                  [140.1, 276.2, 374.4, 1.001], [160.3, 272.3, 378.2, 1.001], [179.7, 269.2, 381.8, 1.001]])
# Array of (Amperage, Lower radius, Upper radius, Voltage)
data2 = np.array(
    [[1.001, 284.5, 366.4, 100.0], [1.102, 288.7, 363.2, 100.6], [1.2, 290.2, 360.4, 100.3], [0.9, 281.4, 371.2, 100.5],
     [0.802, 276.6, 378.0, 100.5], [0.7, 270.8, 384.0, 100.51]])
# Array of (X-pos, Y-pos, Magnetic field)
data3 = np.array(
    [[25.2, 12.5, 0.763], [16.0, 19.0, 0.768], [20.5, 8.5, 0.790], [17.0, 12.0, 0.790], [17.0, 16.5, 0.792],
     [17.0, 20.9, 0.744]])
# Array of (Voltage, Amperage, Magnetic Field)
data4 = np.array([[5.9, 1.29, 0.790], [4.8, 1.04, 0.599], [6.5, 1.41, 0.881], [3.6, 0.78, 0.397], [2.6, 0.587, 0.236],
                  [1.0, 0.24, -0.014], [7.3, 1.58, 1.008], [8.0, 1.74, 1.134], [8.7, 1.89, 1.251], [9.2, 2.00, 1.337]])
earth_mf = -0.00020427507168401274

def combine_weighted(data, data_s):
    weight = 1.0 / np.power(data_s, 2)
    weighted_average = (data * weight).sum() / weight.sum()
    error = (1 / np.sqrt(weight.sum()))
    return (weighted_average, error)


print("E/M values for constant amperage")
# Constant amperage:
magnetic_field_per_i = np.pi * 4 * 10 ** -7 * 0.15 ** 2 / (2 * (0.15 ** 2 + 0.075 ** 2) ** 1.5)
n = 130 * 2
spread_radius = 0.005
radius_spool = 0.15
x = 0.5 * radius_spool
spread_radius_spool = 0.005
data = []
data_s = []
for row in data1:
    diameter = (row[2] - row[1]) * 0.001  # Diameter van de electronstraal
    voltage = row[0]
    amperage: float = row[3]
    spread_amperage = amperage * 0.01 + 0.003
    spread_voltage = 0.007 * voltage + 0.002  # Spreiding van de spanning
    magnetic_field = n * magnetic_field_per_i * amperage - earth_mf * 0.25
    radius = 0.5 * diameter
    spread_x = 0.5 * spread_radius_spool
    e_div_m: float = 2 * voltage / (radius ** 2 * magnetic_field ** 2)  # De berekende waarde van e/m
    spread_magnetic_field = (1 / amperage ** 2 * spread_amperage ** 2 + 4 * spread_radius_spool ** 2 / (
            9 * radius_spool ** 2)) ** 0.5 * magnetic_field
    s_e_div_m: float = (1 / (
            voltage ** 2) * spread_voltage ** 2 + 4 / radius ** 2 * spread_radius ** 2 + 4 / magnetic_field ** 2 * spread_magnetic_field ** 2) ** 0.5 * e_div_m  # De bijbehorende spreiding
    print(
        f"{voltage} $\pm{{ {spread_voltage:.2f} }}$ & {(diameter / 2) * 100:.2f} $\pm{{ {spread_radius:.2f} }}$ & {amperage:.2f} $\pm{{ {spread_amperage:.2f} }}$ & {(e_div_m / 10 ** 11):.2f} $\pm{{ {str(round(s_e_div_m / 10 ** 11, 2))} }}$ \\\\")
    data.append(e_div_m)
    data_s.append(s_e_div_m)
(averageca, errorca) = combine_weighted(data, data_s)
print(f"{averageca / 10 ** 11:.2f}, {errorca / 10 ** 11:.2f}")

print("E/M values for constant voltage")
# Constant voltage:
magnetic_field_per_i = np.pi * 4 * 10 ** -7 * 0.15 ** 2 / (2 * (0.15 ** 2 + 0.075 ** 2) ** 1.5)
n = 130 * 2
spread_radius = 0.005
radius_spool = 0.15
x = 0.5 * radius_spool
spread_radius_spool = 0.005
data = []
data_s = []
for row in data2:
    diameter = (row[2] - row[1]) * 0.001  # Diameter van de electronstraal
    voltage = row[3]
    amperage: float = row[0]
    spread_amperage = amperage * 0.01 + 0.003
    spread_voltage = 0.007 * voltage + 0.002  # Spreiding van de spanning
    magnetic_field = n * magnetic_field_per_i * amperage - earth_mf * 0.25
    radius = 0.5 * diameter
    spread_x = 0.5 * spread_radius_spool
    e_div_m: float = 2 * voltage / (radius ** 2 * magnetic_field ** 2)  # De berekende waarde van e/m
    spread_magnetic_field = (1 / amperage ** 2 * spread_amperage ** 2 + 4 * spread_radius_spool ** 2 / (
            9 * radius_spool ** 2)) ** 0.5 * magnetic_field
    s_e_div_m: float = (1 / (
            voltage ** 2) * spread_voltage ** 2 + 4 / radius ** 2 * spread_radius ** 2 + 4 / magnetic_field ** 2 * spread_magnetic_field ** 2) ** 0.5 * e_div_m  # De bijbehorende spreiding
    print(
        f"{voltage} $\pm{{ {spread_voltage:.2f} }}$ & {(diameter / 2) * 100:.2f} $\pm{{ {spread_radius:.2f} }}$& {amperage:.2f} $\pm{{ {spread_amperage:.2f} }}$ & {(e_div_m / 10 ** 11):.2f} $\pm{{ {str(round(s_e_div_m / 10 ** 11, 2))} }}$ \\\\")
    data.append(e_div_m)
    data_s.append(s_e_div_m)
(average, error) = combine_weighted(data, data_s)
print(f"{average / 10 ** 11:.2f}, {error / 10 ** 11:.2f}")

#plt.rcParams.update({'font.size': 14})

def voltage_fit(r, c):
    return c * r ** 2


# Constant amperage
fit = fitcode.curve_fit(voltage_fit, (data1.transpose()[2] - data1.transpose()[1])*0.001, data1.transpose()[0],
                        spread_radius, data1.transpose()[0] * 0.007 + 0.002, [10])
print("Plot for constant amperage")
xs = np.linspace(0.05, 0.125, 1000)
ys = fit[0] * xs ** 2
theoretical_param = 0.5*(1.602*10**-19/(9.1093837*10**-31))*(4*np.pi*10**-7*0.15**2/(2*(0.15**2+0.075**2)**(3/2)))**2*130**2
print((4*np.pi*10**-7*0.15**2/(2*(0.15**2+0.075**2)**(3/2)))**2)
plt.scatter(data1.transpose()[0], (data1.transpose()[2] - data1.transpose()[1])*0.001)
plt.errorbar(data1.transpose()[0], (data1.transpose()[2] - data1.transpose()[1])*0.001,
             xerr=data1.transpose()[0] * 0.007 + 0.002, yerr=spread_radius, fmt='o')
plt.plot(ys,xs)
plt.ylabel("Electronenstraal (m)")
plt.xlabel("Spanning (V)")
plt.plot(theoretical_param * xs ** 2, xs)
plt.legend(["Gemeten datapunten", "Plot van gefitte functie", "Theoretische waarde"])
plt.savefig("constantamperage.png")
plt.show()
print(theoretical_param, fit[0])


def amperage_fit(r, c):
    return c / r


fit = fitcode.curve_fit(amperage_fit, (data2.transpose()[2] - data2.transpose()[1])*0.001, data2.transpose()[0],
                        data2.transpose()[0] * 0.01 + 0.003, spread_radius, [10])
theoretical_param = (2 * 100 * 9.1093837*10**-31/(1.602*10**-19*130**2*(4*np.pi*10**-7*0.15**2/(2*(0.15**2+0.075**2)**(3/2)))**2)) ** 0.5
print("Plot for constant voltage")
xs = np.linspace(0.05, 0.2, 1000)
ys = fit[0] / xs
plt.scatter(data2.transpose()[0], (data2.transpose()[2] - data2.transpose()[1])*0.001)
plt.errorbar(data2.transpose()[0], (data2.transpose()[2] - data2.transpose()[1])*0.001,
             xerr=data2.transpose()[0] * 0.01 + 0.003, yerr=spread_radius, fmt='o')
plt.plot(ys, xs)
plt.plot(theoretical_param / xs, xs)
plt.ylabel("Electronenstaal (m)")
plt.xlabel("Stroom (A)")
plt.legend(["Gemeten datapunten", "Plot van de fefitte functie", "Theoretische waarde"])
plt.savefig("constantvoltage.png")
plt.show()

print("Plot for distance from center compared to the magnetic field strength")


def dist_to_mid(x, y):
    return ((x-0.17)**2+(y-0.12)**2) ** 0.5


plt.scatter(dist_to_mid(data3.transpose()[0]*0.01,data3.transpose()[1]*0.01), data3.transpose()[2])
plt.xlabel("Afstand tot het middelpunt (m)")
plt.ylabel("Magnetische veldsterkte (mT)")
plt.legend(["Gemeten veldsterktes"])
plt.savefig("fieldstrength.png")
plt.show()
b_div_i = 0.79 / 1.29

def fit_magnetic_field(i, a, b):
    return a * i + b


print("Plot for The magnetic field in the center compared to the amperage")

fit = fitcode.curve_fit(fit_magnetic_field, data4.transpose()[1], data4.transpose()[2], data4.transpose()[1] * 0.01 + 0.003, 0.01 ,[1.0, 0.0])
xs = np.linspace(0, 2, 1000)
ys = fit[0][0] * xs + fit[0][1]
plt.scatter(data4.transpose()[1],data4.transpose()[2])
plt.plot(xs,ys)
plt.xlabel("Stroom(A)")
plt.ylabel("Magneetveld(mT)")
plt.legend(["Gemeten datapunten","Gefitte functie"])
plt.savefig("earthmagneticfield.png")
print(f"Aardmagnetisch veld: ~{fit[0][1]} $\pm{{ {fit[1][1]} }}$")
plt.show()
for row in data4:
    print(f"{row[0]:.2f} $\pm{{ {(0.007 * row[0] + 0.002):.2f} }}$ & {row[1]:.2f} $\pm{{ {(row[1]*0.01+0.003):.2f} }}$ & {row[2]:.2f} $\pm{{0.01}}$ \\\\")
print(average,averageca,error,errorca)
print(combine_weighted(np.array([averageca,average]), np.array([errorca,error])))