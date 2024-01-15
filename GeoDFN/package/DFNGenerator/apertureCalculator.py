import math

# LinearLengthAperture is a version of sub linear considering the calculation of the coefficient based on rock properties. we excluded this as it was considered some extera work for users to provide this data
class LinearLengthAperture:
    def __init__(self, fractures, apertureCalculationParameters,stage):
        self.stage=stage
        self.fractures = fractures
        self.K_Ic = apertureCalculationParameters["K_Ic"]  # Fracture toughness in MPa√m
        self.E = apertureCalculationParameters["E"]/(10e6)        # Young's modulus in MPa
        self.v = apertureCalculationParameters["nu"]        # Poisson's ratio (dimensionless)
        self.exponent = apertureCalculationParameters["scaling_exponent"]
        self.calculate_apertures()
    def calculate_apertures(self):
        for frac in self.fractures:
            C = self.calculate_scaling_parameter(self.K_Ic, self.E, self.v)
            d_max = C * (frac['fracture length'] ** self.exponent)
            if self.stage=='first':
                frac['fracture aperture'] = d_max
            else:
                frac['correctedAperture' + str(self.strike)] = d_max
        return self.fractures

    def calculate_scaling_parameter(self, K_Ic, E, v):
        # Convert E from GPa to MPa by multiplying by 1000 because K_Ic is in MPa√m
        E_MPa = E * 1000
        C = (K_Ic * (1 - v ** 2)) / (E_MPa / (2 * math.pi) ** 0.5)
        return C


class subLinearLengthAperture:
    def __init__(self, fractures, apertureCalculationParameters,stage):
        self.stage=stage
        self.fractures = fractures
        self.scalingCoefficient = apertureCalculationParameters["scalingCoefficient"]  # Fracture toughness in MPa√m
        self.exponent = apertureCalculationParameters["scalingExponent"]
        self.calculate_apertures()
    def calculate_apertures(self):
        for frac in self.fractures:
            d_max = self.scalingCoefficient * (frac['fracture length'] ** self.exponent)
            if self.stage=='first':
                frac['fracture aperture'] = d_max
            else:
                frac['correctedAperture' + str(self.strike)] = d_max
        return self.fractures


class BisdomAperture:
    def __init__(self, fractures, apertureCalculationParameters,stage):
        self.stage=stage
        self.fractures = fractures
        self.JRC = apertureCalculationParameters["JRC"]
        self.JCS = apertureCalculationParameters["JCS"]
        self.S_Hmax = apertureCalculationParameters["sigma_Hmax"]
        self.strike = apertureCalculationParameters["strike"]
        self.S_C=apertureCalculationParameters["sigma_c"]

        self.calculate_apertures()

    def calculate_apertures(self):
        E_0 = self.calculate_E0(self.JRC, self.JCS, self.S_C)
        V_m = self.calculate_Vm(E_0, self.JRC, self.JCS)
        K_ni = self.calculate_Kni(E_0, self.JRC, self.JCS)
        for frac in self.fractures:
            alpha = abs(frac['theta'] - self.strike)
            sigma_n_single_fracture = self.S_Hmax * (-0.33 * math.cos(math.radians(alpha)) + 0.65)
            sigma_n_length = sigma_n_single_fracture * (-0.083 * math.log(frac['fracture length']) + 1.055)


            E_n = E_0 -( (1 / V_m) + (K_ni / sigma_n_length)) ** (- 1)
            #print("E_n= ", E_n)
            # Ensure the aperture does not go negative
            corrected_aperture = max(E_n, 0)
            if self.stage=='first':
                frac['fracture aperture'] = corrected_aperture
                frac['initial aperture']=E_0
            else:
                frac['correctedAperture' + str(self.strike)] = corrected_aperture
        return self.fractures

    def calculate_E0(self, JRC, JCS,sigma_c):
        return (JRC / 5) * ((0.2 * sigma_c / JCS) - 0.1)

    def calculate_Vm(self, E_0, JRC, JCS):
        return -0.1032 - (0.0074 * JRC) + 1.135 * ((JCS / (E_0 * 10))**-0.251)

    def calculate_Kni(self, E_0, JRC, JCS):
        return -7.15 + 1.75 * JRC + 0.02 * (JCS / E_0)

class constantAperture:
    def __init__(self, fractures, apertureCalculationParameters,stage):
        self.stage=stage
        self.fractures = fractures
        self.aperture=apertureCalculationParameters["aperture"]
    def calculate_apertures(self):
        for frac in self.fractures:
            if self.stage=='first':
                frac['fracture aperture'] = self.aperture
            else:
                frac['correctedAperture' + str(self.strike)] = self.aperture
        return self.fractures

class LepillierAperture:
    def __init__(self, fractures, apertureCalculationParameters,stage):
        self.stage=stage
        self.temporaryFractures = constantAperture(fractures, apertureCalculationParameters,stage='first').calculate_apertures()
        self.strike=apertureCalculationParameters['strike']
        # Extracting rock properties from the dictionary based on rockType
        self.E_matrix = apertureCalculationParameters["E"]
        self.nu_matrix = apertureCalculationParameters["nu"]
        self.S_Hmax=apertureCalculationParameters["S_Hmax"]
        self.S_hmin=apertureCalculationParameters["S_hmin"]
        self.fractures = fractures
        self.compute_kn(self.fractures)
        #calculating the stress

    def calculate_apertures(self):
        i=0
        for frac in self.fractures:
            sigma_n = self.stress_decomposition(self.S_Hmax,  self.S_hmin, self.strike, frac['theta'])
            diff=sigma_n/frac['kn']
            newAperture=self.temporaryFractures[i]['fracture aperture']-diff
            i=+1
            if newAperture<0:
                newAperture=0
            if self.stage=='first':
                frac['initial aperture']=self.temporaryFractures[i]['fracture aperture']
                frac['fracture aperture'] = newAperture
            else:
                frac['correctedAperture' + str(self.strike)] = newAperture
        return self.fractures
            #frac['correctedAperture'+str(self.stressAzimuth)]=newAperture



    def compute_kn(self, fractures):
        """Compute the normal stiffness or spring constant."""
        E_fracture = 0.1 * self.E_matrix
        nu_fracture = 0.4 * self.nu_matrix
        for frac in fractures:
            frac['kn'] = E_fracture * (1 - nu_fracture) / (frac['fracture aperture'] * (1 + nu_fracture) * (1 - 2 * nu_fracture))

    def stress_decomposition(self, S_Hmax, S_hmin, stress_azimuth, fracture_orientation):
        # Calculating the difference between stress azimuth and fracture orientation
        delta_theta = abs(stress_azimuth - fracture_orientation)
        # Calculating the normal stress (σn) on the fracture
        sigma_n = S_Hmax * math.sin(math.radians(delta_theta)) ** 2 + \
                  S_hmin * math.cos(math.radians(delta_theta)) ** 2

        return sigma_n #to convert to Pa



class apertureCalculator:
    def __init__(self, apertureCalculationParameters,stage='first'):
        self.apertureCalculationParameters=apertureCalculationParameters
        self.method=apertureCalculationParameters["method"]
        self.stage=stage
    def get_calculator(self, fractures):
        if self.method == 'subLinear':
            return subLinearLengthAperture(fractures, self.apertureCalculationParameters,self.stage).calculate_apertures()
        elif self.method=='constant':
            return constantAperture(fractures, self.apertureCalculationParameters,self.stage).calculate_apertures()
        elif self.method == 'Bisdom':
            return BisdomAperture(fractures, self.apertureCalculationParameters,self.stage).calculate_apertures()#, S_Hmax, orientation_of_SHmax
        elif self.method == 'Lepillier':
            return LepillierAperture(fractures, self.apertureCalculationParameters, self.stage).calculate_apertures()
        else:
            raise ValueError("Unknown method for aperture calculation")


