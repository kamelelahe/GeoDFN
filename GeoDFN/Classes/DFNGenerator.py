import random
import json
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import math
import os
from .apertureCalculator import apertureCalculator
from .bufferZoneCalculator import bufferZoneCalculator
from .fractureLengthPDFs import fractureLengthPDFs
from .spatialDisturbutionPDFs import spatialDisturbutionPDFs
from .orienetationPDFs import orientationPDFs

import matplotlib.colors as mcolors


class DFNGenerator:

    def __init__(self, domainLengthX, domainLengthY, sets, apertureCalculationParameters, DFNName,numOfRealizations=1,
                 IsMultipleStressAzimuths=False, stressAzimuth=None, savePic=True):
        """
        Initializes the DFNGenerator class to create Discrete Fracture Networks (DFNs) based on provided configurations.

        Parameters:
        -----------
        domainLengthX : float
            The length of the simulation domain in the X direction.
            Example: `domainLengthX = 300`

        domainLengthY : float
            The length of the simulation domain in the Y direction.
            Example: `domainLengthY = 600`

        sets : list of dicts
            A list of dictionaries, each defining a fracture set's properties.
            Each dictionary should include keys such as 'I' (intensity), 'fractureLengthPDF',
            'fractureLengthPDFParams', 'spatialDisturbutionPDF', 'spatialDisturbutionPDFParams',
            'orientationDisturbutonPDF', 'orientationDisturbutonPDFParams', and 'bufferZone'.

            Example:
            ```python
            set1 = {
                'I': 0.01,
                'fractureLengthPDF': "Log-Normal",
                'fractureLengthPDFParams': {"mu": 2.4, "sigma": 0.73, "Lmin": 2.59, "Lmax": 57.48},
                'spatialDisturbutionPDF': "Power-law",
                'spatialDisturbutionPDFParams': {"alpha": 0.51, "min distance": 1, "max distance": 600},
                'orientationDisturbutonPDF': "Von-Mises",
                'orientationDisturbutonPDFParams': {"kappa": 8.55 , "loc": 1.4, 'thetaMin': np.radians(30), 'thetaMax': np.radians(120)},
                'bufferZone': {"constant": 1.4, "method": "constant"}
            }
            ```

        apertureCalculationParameters : dict
            Parameters specifying how to calculate fracture apertures.
            'constant', 'subLinear', ''Barton-Bandis'', and 'Lepillier' are availabke

            Example:
            ```python
            apertureCalculationParameters = {
                "method": 'subLinear',
                "scalingCoefficient": 0.001,
                "scalingExponent": 0.5,
            }
            ```

        DFNName : str
            The name of the DFN, used for naming the output directories and files.
            Example: `DFNName = 'MyDFN'`

        numOfRealizations : int, optional (default=1)
            Number of DFN realizations to generate.
            Each realization will follow the specified fracture sets and aperture parameters but with different random placements.

        IsMultipleStressAzimuths : bool, optional (default=False)
            If True, the aperture calculation will consider multiple stress azimuths.
            This can model the effect of varying stress orientations on the fracture apertures.

        stressAzimuth : list of floats, optional
            List of stress azimuth angles to consider when calculating fracture apertures if IsMultipleStressAzimuths is True.
            Example: `stressAzimuth = [0, 45, 90]`

        savePic : bool, optional (default=True)
            If True, the class will save visualizations of the DFN for each realization.
            The images include the fracture network, stress directions, and stereographic projections of orientations.

        Usage Example:
        --------------
        ```python
        DFNGenerator(domainLengthX=300,
                     domainLengthY=600,
                     sets=[set1, set2, set3],
                     apertureCalculationParameters=apertureCalculationParameters,
                     DFNName='12Brazil-forPaper',
                     numOfRealizations=10,
                     IsMultipleStressAzimuths=False,
                     stressAzimuth=[0, 45, 90])
        ```

        Notes:
        ------
        - The class is used for generating DFNs.
        - The fractures are generated, sorted, placed within the domain, and their apertures are calculated.
        - If multiple stress azimuths are used, the aperture of each fracture is recalculated for each azimuth.
        - Outputs include text files with fracture properties, coordinates, and images visualizing the fracture network.
        """
        self.maxtries = []
        ##  initialization of disturbution functions
        self.xmax = domainLengthX
        self.ymax = domainLengthY
        outputDir = 'DFNs/' + str(DFNName)
        self.outputDir = outputDir
        self.apertureCalculation = apertureCalculator(apertureCalculationParameters, stage='first')
        self.numberOfMaxTries = 400000
        NumOfRealization = numOfRealizations
        for i in range(NumOfRealization):
            ###############################################################################
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('+++++++ Section A: generate  fractures +++++++++')
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            self.maxtries = []
            allFractureSets = []
            self.setNumber=1
            for setConfig in sets:
                fractureSet, setConfig = self.generateFractures(setConfig)
                fractureSet = self.sortFractures(fractureSet)
                allFractureSets.append((fractureSet, setConfig))
                self.setNumber =self.setNumber + 1

            ###############################################################################
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('+++++++ Section B: placing the fractures +++++++++')
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

            allProcessedFractureSets = []
            for fractureSet, setConfig in allFractureSets:
                print(setConfig['bufferZone'])
                self.bufferZoneCalculation = bufferZoneCalculator(setConfig['bufferZone'])
                fractureSet = self.bufferZoneCalculation.calculate(fractureSet)

                # Now you can use set_config['spatialDisturbutionPDF'] and set_config['spatialDisturbutionPDFParams']
                processedFractureSet = self.place_fractures(fractureSet, setConfig)
                allProcessedFractureSets.append(processedFractureSet)

            print('maxtries=', self.maxtries)

            if  self.maxtries[0]>self.numberOfMaxTries or self.maxtries[1]>self.numberOfMaxTries:
                maxtriesDir = os.path.join(self.outputDir, 'tries')
                os.makedirs(maxtriesDir, exist_ok=True)
                maxtriesFile = os.path.join(maxtriesDir, 'tries.txt')
                with open(maxtriesFile, 'w') as fileID:
                    fileID.write(f"Number of iterations for each set: {self.maxtries}\n")

            else:
                maxtriesDir = os.path.join(self.outputDir, 'tries')
                os.makedirs(maxtriesDir, exist_ok=True)
                maxtriesFile = os.path.join(maxtriesDir, 'tries.txt')
                with open(maxtriesFile, 'w') as fileID:
                    fileID.write(f"Number of iterations for each set: {self.maxtries}\n")
                ###############################################################################
                # this sub section deals with calculation realted to influence of stress field on aperture
                if IsMultipleStressAzimuths:
                    self.stressAzimuth = stressAzimuth
                    for azimuth in stressAzimuth:
                        apertureCalculationParameters["strike"] = azimuth
                        self.apertureCalculation = apertureCalculator(apertureCalculationParameters, stage='second')
                        for set in allProcessedFractureSets:
                            correlatedSet = self.apertureCalculation.get_calculator(set)
                            set = correlatedSet

                ###############################################################################
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                print('+++++++ Section D: Generating the outputs +++++++++')
                print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

                if savePic:
                    self.plotFractures(allProcessedFractureSets, name='DFNPic', number=i)
                    #self.plotFracturesWithAperture(allProcessedFractureSets, name='DFNPicAperture', number=i)

                self.generateOutputFileForOutputPropertiesPerSet('outputPropertiesPerSet', allProcessedFractureSets,
                                                                 number=i)
                self.generateOutputFileForOverallProperties('outputPropertiesTotal', allProcessedFractureSets, number=i)
                self.generateTextFileForFractureCoordinates('fractureCoordinates', allProcessedFractureSets, number=i)
                self.generateTextFileForFractureSet('fractureSet', allProcessedFractureSets, number=i)
                self.generateTextFileForFractureApertures('aperture', allProcessedFractureSets, number=i)
                self.generateInputPropertiesFile('inputProperties',apertureCalculationParameters, sets, stressAzimuth, number=i)
                self.plotOrientationStereographic('orientationStereographic', allProcessedFractureSets, number=i)
                if IsMultipleStressAzimuths:
                    self.generateTextFilesForCorrectedApertures('correlatedAperture', allProcessedFractureSets, number=i)
                    self.plotCorrectedApertures('aperturePerStrikeTotal', allProcessedFractureSets, number=i)


    def placeLongestFracture(self, longestFractrue):
        """
        Place the longest fracture within the domain, ensuring it fits within boundaries.

        Parameters:
        -----------
        longestFractrue : dict
            The longest fracture to be placed, including properties like length, spacing, and orientation.

        Returns:
        --------
        tuple
            A tuple containing the placed fracture with updated coordinates and the seed point (x, y) used for placement.
            Returns False if the fracture cannot be placed within the maximum number of tries.
        """
        referenceWithinDomain = False
        number = 0
        tries = 0
        print('fracture length', longestFractrue['fracture length'])
        while not referenceWithinDomain:
            seed_x = random.uniform(0, self.xmax)
            seed_y = random.uniform(0, self.ymax)
            #print(self.fractureCoordinate(longestFractrue, longestFractrue['theta'], seed_x, seed_y))
            (new_x_start, new_y_start), (new_x_end, new_y_end) = self.fractureCoordinate(longestFractrue,
                                                                                         longestFractrue['theta'],
                                                                                         seed_x, seed_y)
            if self.is_within_domain(new_x_start, new_y_start) and self.is_within_domain(new_x_end, new_y_end):
                referenceWithinDomain = True
                addedFracture = {
                    'number': number,
                    'x_start': new_x_start,
                    'y_start': new_y_start,
                    'x_end': new_x_end,
                    'y_end': new_y_end,
                    'fracture length': longestFractrue['fracture length'],
                    'set number': longestFractrue['set number'],
                    'fracture spacing': longestFractrue['fracture spacing'],
                    'theta': longestFractrue['theta'],
                }
            else:
                tries = +1
                if tries > 500:
                    return False

        return (addedFracture, seed_x, seed_y)

    def place_fractures(self, fractures, setConfig):
        """
        Place fractures in the domain based on spatial distribution and domain constraints.

        Parameters:
        -----------
        fractures : list
            List of fractures to be placed in the domain.

        setConfig : dict
            Configuration settings for fracture placement, including spatial distribution.

        Returns:
        --------
        list
            Processed fractures with updated coordinates and properties.
        """
        spatialDisturbutionPDF = spatialDisturbutionPDFs(setConfig['spatialDisturbutionPDF'],
                                                         setConfig['spatialDisturbutionPDFParams'])
        spatialDisturbutionPDFMode = spatialDisturbutionPDF.compute_mode()

        # Step A2: Fracture Placement
        processedFractures = []
        print('--- placing the longest fracture----')
        starting = 0
        added = False
        while not added:
            added = self.placeLongestFracture(fractures[0])

        (addedFracture, seed_x, seed_y) = added

        processedFractures.append(addedFracture)
        print('--the longest fracture added--')
        numberOfTries = 0
        maxTriesReached = False

        number = 0
        # Introducing New Fractures
        print('--- placing rest of fractures ---')
        for fracture in fractures[1:]:
            theta = fracture['theta']
            if theta < 0:
                theta += 360
            if maxTriesReached:  # Check if max tries have been reached before processing a new fracture
                break
            isNewFractureAdded = False

            while not isNewFractureAdded:
                if numberOfTries > self.numberOfMaxTries:  # Global maximum tries check
                    print("Global max retries reached, stopping all fracture placements.")
                    maxTriesReached = True  # Set the flag to indicate maximum tries have been reached
                    break  # Break out of the while loop

                referenceWithinDomain = False
                while not referenceWithinDomain:
                    if numberOfTries > self.numberOfMaxTries:  # Check again in case the threshold is reached in this inner loop
                        print("Global max retries reached during domain validation, stopping all fracture placements.")
                        maxTriesReached = True
                        break

                    distance = (spatialDisturbutionPDF.get_value() - spatialDisturbutionPDFMode) * random.choice(
                        [-1, 1])  # Randomness for + and -

                    angle = np.random.uniform(0, 2 * np.pi)  # Random angle for 2D normal distribution

                    new_x_mid = seed_x + distance * np.cos(angle)
                    new_y_mid = seed_y + distance * np.sin(angle)

                    (new_x_start, new_y_start), (new_x_end, new_y_end) = self.fractureCoordinate(fracture, theta,
                                                                                                 new_x_mid, new_y_mid)
                    # add check proximity here
                    if self.is_within_domain(new_x_start, new_y_start) and self.is_within_domain(new_x_end, new_y_end):
                        referenceWithinDomain = True
                    else:
                        numberOfTries += 1
                        #print("COORDINATE: fracture number", str(fracture['number']), 'with length of ',
                        #      str(fracture['fracture length']), ' has been relocated. I= ',
                        #      str(self.computeIntensity(processedFractures)), "has been reached", str(numberOfTries),
                        #      "times")
                # Check proximity
                if maxTriesReached:  # Check the flag after exiting the inner loop
                    break
                too_close = False
                for existing_frac in processedFractures:
                    existing_coords = ((existing_frac['x_start'], existing_frac['y_start']),
                                       (existing_frac['x_end'], existing_frac['y_end']))
                    new_coords = ((new_x_start, new_y_start), (new_x_end, new_y_end))
                    if segment_to_segment_distance(new_coords, existing_coords) < existing_frac['fracture spacing'] + \
                            fracture['fracture spacing']:
                        too_close = True
                        numberOfTries += 1
                        #print("BUFFERZONE: fracture number", str(fracture['number']), 'with length of ',
                        #      str(fracture['fracture length']), ' has been relocated. I= ',
                        #      str(self.computeIntensity(processedFractures)), "has been reached", str(numberOfTries),
                        #      "times")

                        break

                if numberOfTries > self.numberOfMaxTries:  # Global maximum tries check
                    print("Global max retries reached, stopping all fracture placements.")
                    maxTriesReached = True  # Set the flag to indicate maximum tries have been reached
                    break  # Break out of the while loop
                if not too_close:
                    isNewFractureAdded = True
                    number += 1
                    addedFracture = {
                        'number': number,
                        'x_start': new_x_start,
                        'y_start': new_y_start,
                        'x_end': new_x_end,
                        'y_end': new_y_end,
                        'fracture length': fracture['fracture length'],
                        'fracture spacing': fracture['fracture spacing'],
                        'set number': fracture['set number'],
                        'theta': fracture['theta'],
                        'fracture aperture':0.08,
                        'number of tries': numberOfTries,
                    }
                    processedFractures.append(addedFracture)
            if maxTriesReached:  # Check the flag after exiting the inner loop
                break

        self.maxtries.append(numberOfTries)

        # this i when we add fractures aperture
        processedFractures = self.apertureCalculation.get_calculator(processedFractures)

        return processedFractures

    def generateFractures(self, setConfig):

        """
        Generate a list of fractures based on the provided configuration.

        Parameters:
        -----------
        setConfig : dict
            Configuration for the fracture set, including distribution and intensity.

        Returns:
        --------
        tuple
            A list of fractures, each with properties like length and orientation, and the updated setConfig.
        """
        orientationDisturbutonPDF = orientationPDFs(setConfig['orientationDisturbutonPDF'],
                                                    setConfig['orientationDisturbutonPDFParams'])

        print('fractureLengthPDFParams["Lmax"]=', setConfig['fractureLengthPDFParams']["Lmax"])

        if setConfig['fractureLengthPDF'] == 'Constant':
            print('Fracture length PDF is constant')
            n = math.ceil(setConfig['I'] * self.ymax * self.xmax / setConfig['fractureLengthPDFParams']['L'])
            fractures = []
            for _ in range(n):
                newFrac = {'fracture length': setConfig['fractureLengthPDFParams']['L']}
                newFrac['theta'] = orientationDisturbutonPDF.get_value()
                newFrac['set number'] = self.setNumber
                fractures.append(newFrac)
        else:
            fractureLengthPDF = fractureLengthPDFs(setConfig['fractureLengthPDF'], setConfig['fractureLengthPDFParams'])
            fractures = []
            newFrac = {}
            newFrac['fracture length'] = fractureLengthPDF.get_value()
            newFrac['theta'] = orientationDisturbutonPDF.get_value()
            newFrac['set number'] = self.setNumber

            fractures.append(newFrac)
            while self.computeIntensity(fractures) < setConfig['I']:
                newFrac = {}
                newFrac['fracture length'] = fractureLengthPDF.get_value()
                newFrac['theta'] = orientationDisturbutonPDF.get_value()
                newFrac['set number'] = self.setNumber

                fractures.append(newFrac)

        return fractures, setConfig

    def computeIntensity(self, fractures):
        """
        Compute the intensity of the fracture network.

        Parameters:
        -----------
        fractures : list
            List of fractures, each with a 'fracture length'.

        Returns:
        --------
        float
            Intensity of the fracture network, defined as total fracture length per unit area.
        """
        total_length = sum([fracture['fracture length'] for fracture in fractures])
        area = self.xmax * self.ymax
        intensity = total_length / area
        return intensity

    def sortFractures(self, fractures):
        """
        Sort fractures by length in descending order and assign a sequence number.

        Parameters:
        -----------
        fractures : list
            List of fractures, each with a 'fracture length'.

        Returns:
        --------
        list
            The sorted list of fractures, with each fracture assigned a sequence number.
        """
        fractures.sort(key=lambda x: x['fracture length'], reverse=True)
        for i, fracture in enumerate(fractures, start=0):
            fracture['number'] = i
        return fractures

    def fractureCoordinate(self, fracture, theta, midX, midY):
        """
        Calculate the start and end coordinates of a fracture given its midpoint and orientation.

        Parameters:
        -----------
        fracture : dict
            Fracture data containing the 'fracture length'.

        theta : float
            Orientation angle of the fracture in degrees.

        midX : float
            X-coordinate of the fracture's midpoint.

        midY : float
            Y-coordinate of the fracture's midpoint.

        Returns:
        --------
        tuple
            Coordinates of the fracture's start and end points as ((x_start, y_start), (x_end, y_end)).
        """
        half_length = fracture['fracture length'] / 2
        theta_adjusted = 90 - theta
        new_x_start = midX - half_length * np.cos(np.radians(theta_adjusted))
        new_y_start = midY - half_length * np.sin(np.radians(theta_adjusted))
        new_x_end = midX + half_length * np.cos(np.radians(theta_adjusted))
        new_y_end = midY + half_length * np.sin(np.radians(theta_adjusted))
        return (new_x_start, new_y_start), (new_x_end, new_y_end)

    def is_within_domain(self, x, y):
        """
        Check if a point (x, y) is within the domain boundaries.

        Parameters:
        -----------
        x : float
            X-coordinate of the point.

        y : float
            Y-coordinate of the point.

        Returns:
        --------
        bool
            True if the point is within the domain, False otherwise.
        """
        return 0 <= x <= self.xmax and 0 <= y <= self.ymax

    #### ==========================================
    #### Functions for Generating Outputs
    #### ==========================================

    def distanceOfWellFromClosetFracture(self, fractures, wellLocation):
        """
        Calculate the shortest distance from a well to the nearest fracture.

        Parameters:
        -----------
        fractures : list
            List of fractures, each containing start and end coordinates.

        wellLocation : tuple
            Coordinates of the well as (x, y).

        Returns:
        --------
        float
            The shortest distance from the well to the nearest fracture.
        """
        min_distance = float('inf')
        for frac in fractures:
            x_start, y_start = frac['x_start'], frac['y_start']
            x_end, y_end = frac['x_end'], frac['y_end']
            distance = point_to_segment_distance(np.array(wellLocation), np.array([x_start, y_start]),
                                                 np.array([x_end, y_end]))
            min_distance = min(min_distance, distance)
        return min_distance

    def generateInputPropertiesFile(self, name, apertureCalculationParameters,fractureSets, stressAzimuth, number=0):
        """
        Generate a text file with input properties for the fracture network simulation.

        Parameters:
        -----------
        name : str
            Name for the output file.

        apertureCalculationParameters : dict
            Parameters used for calculating fracture apertures.

        fractureSets : list
            List of fracture sets, each containing fracture properties.

        stressAzimuth : list
            List of stress azimuth angles used in the simulation.

        number : int, optional
            Realization number for file naming (default is 0).

        Notes:
        ------
        - Saves domain dimensions, fracture sets, stress azimuths, and aperture calculation parameters to a text file.
        - The file is saved in the specified output directory.
        """
        InputPropertiesDir = os.path.join(self.outputDir, name)
        os.makedirs(InputPropertiesDir, exist_ok=True)
        InputPropertiesFile = os.path.join(InputPropertiesDir, f"{number + 1:03}{name}.txt")
        with open(InputPropertiesFile, 'w') as f:
            f.write("domainLengthX: " + str(self.xmax) + '\n')
            f.write("domainLengthY: " + str(self.ymax) + '\n')

            i = 1
            for set in fractureSets:
                if isinstance(set, dict):
                    f.write(
                        "fractureSet " + str(i) + " : " + json.dumps(set, indent=4) + '\n')
                else:
                    f.write("fractureSet " + str(i) + " : " + '\n')
                i += 1

            f.write("stressAzimuth: " + str(stressAzimuth) + '\n')
            f.write(
                "fracture aperture " + str(i) + " : " + json.dumps(apertureCalculationParameters, indent=4) + '\n')

    def generateTextFileForFractureCoordinates(self, name, fractures, number=0):
        """
        Generate a text file with the coordinates of fracture endpoints.

        Parameters:
        -----------
        name : str
            Name for the output file.

        fractures : list
            List of fracture sets, each containing fractures with coordinate data.

        number : int, optional
            Realization number for file naming (default is 0).

        Notes:
        ------
        - Saves the start and end coordinates of each fracture in a text file in the specified output directory.
        """
        coordinatesOutputDir = os.path.join(self.outputDir, name)
        os.makedirs(coordinatesOutputDir, exist_ok=True)
        InputPropertiesFile = os.path.join(coordinatesOutputDir, f"{number + 1:03}{name}.txt")

        with open(InputPropertiesFile, 'w') as fileID:
            for set in fractures:
                for frac in set:
                    fileID.write(
                        f"{frac['x_start']:.4f} {frac['y_start']:.4f} {frac['x_end']:.4f} {frac['y_end']:.4f}\n")

    def generateTextFileForFractureSet(self, name, fractures, number=0):
        """
        Generate a text file with the set number for each fracture.

        Parameters:
        -----------
        name : str
            Name for the output file.

        fractures : list
            List of fracture sets, each containing fractures with set number data.

        number : int, optional
            Realization number for file naming (default is 0).

        Notes:
        ------
        - Saves the set number of each fracture in a text file in the specified output directory.
        """
        setOutputDir = os.path.join(self.outputDir, name)
        os.makedirs(setOutputDir, exist_ok=True)
        InputPropertiesFile = os.path.join(setOutputDir, f"{number + 1:03}{name}.txt")

        with open(InputPropertiesFile, 'w') as fileID:
            for set in fractures:
                for frac in set:

                    fileID.write(
                        f"{frac['set number']}\n")

    def generateTextFileForFractureApertures(self, name, fractures, number=0):
        """
        Generate a text file listing the fracture apertures.

        Parameters:
        -----------
        name : str
            Name for the output file.

        fractures : list
            List of fracture sets, each containing fractures with aperture data.

        number : int, optional
            Realization number for file naming (default is 0).

        Notes:
        ------
        - Saves the apertures for all fractures in a text file in the specified output directory.
        """
        apertureOutputDir = self.outputDir + '/' + str(name)
        apertureOutputDir = os.path.join(self.outputDir, name)
        os.makedirs(apertureOutputDir, exist_ok=True)
        InputPropertiesFile = os.path.join(apertureOutputDir, f"{number + 1:03}{name}.txt")

        with open(InputPropertiesFile, 'w') as fileID:
            for set in fractures:
                for frac in set:
                    fileID.write(f"{frac['fracture aperture']:.7f}\n")

    def generateTextFilesForCorrectedApertures(self, name, fractures, number=0):
        """
        Generate separate text files for each corrected aperture.

        Parameters:
        -----------
        name : str
            Name for the output files.

        fractures : list
            List of fracture sets, each containing fractures with corrected aperture data.

        number : int, optional
            Realization number for file naming (default is 0).

        Notes:
        ------
        - Creates a text file for each corrected aperture, storing values for all fractures.
        - Files are saved in the specified output directory, organized by stress azimuth.
        """
        # Create the main directory for corrected apertures if it doesn't exist
        correctedApertureDir = os.path.join(self.outputDir, str(name))
        os.makedirs(correctedApertureDir, exist_ok=True)

        # Initialize a dictionary to hold file handles
        file_handles = {}

        try:
            for set_index, fracture_set in enumerate(fractures):
                for fracture in fracture_set:
                    for key, value in fracture.items():
                        if key.startswith('correctedAperture'):
                            # Create a new file for each corrected aperture if it doesn't exist
                            if key not in file_handles:
                                filePath = os.path.join(correctedApertureDir, f"{key}_{number + 1:03}.txt")
                                file_handles[key] = open(filePath, 'w')

                            # Write the aperture value to the corresponding file
                            file_handles[key].write(f"{value:.7f}\n")
        finally:
            # Close all the file handles
            for file in file_handles.values():
                file.close()

    def generateOutputFileForOutputPropertiesPerSet(self, name, fractureSets, number=0):
        """
        Generate a text file with detailed properties for each fracture set.

        Parameters:
        -----------
        name : str
            Name for the output file.

        fractureSets : list
            List of fracture sets containing fracture data.

        number : int, optional
            Realization number for file naming (default is 0).

        Notes:
        ------
        - Calculates and writes properties such as intensity, length, and aperture statistics for each set.
        - Includes statistics for corrected apertures based on stress azimuths.
        - Saves the results in a text file in the specified output directory.
        """
        outputFileForOutputPropertiesDir = os.path.join(self.outputDir, name)
        os.makedirs(outputFileForOutputPropertiesDir, exist_ok=True)
        outputPropertiesFile = os.path.join(outputFileForOutputPropertiesDir, f"{number + 1:03}{name}.txt")

        with open(outputPropertiesFile, 'w') as fileID:
            for i, fractures in enumerate(fractureSets):
                # Calculate intensity, lengths, apertures, etc., for each set
                intensity = self.computeIntensity(fractures)
                lengths = [frac['fracture length'] for frac in fractures]
                apertures = [frac['fracture aperture'] for frac in fractures]

                # Calculate statistics for lengths and apertures
                minL = min(lengths)
                maxL = max(lengths)
                avgL = sum(lengths) / len(lengths)
                minAperture = min(apertures)
                maxAperture = max(apertures)
                avgAperture = sum(apertures) / len(apertures)

                # Store properties
                setProperties = {
                    'intensity': intensity,
                    'minLength': minL,
                    'maxLength': maxL,
                    'avgLength': avgL,
                    'minAperture': minAperture,
                    'maxAperture': maxAperture,
                    'avgAperture': avgAperture
                }
                fileID.write(f"Properties for set{i + 1}:\n")
                for key, value in setProperties.items():
                    if value > 1:
                        fileID.write(f"{key} : {value:.3f}\n")  # Round for values larger than 1
                    else:
                        fileID.write(f"{key} : {value:.3e}\n")  # Scientific notation otherwise
                # Calculate and write correctedAperture values for the current set
                stressAzimuths = set()
                for frac in fractures:
                    for key in frac:
                        if key.startswith('correctedAperture'):
                            stressAzimuths.add(key)

                for azimuth in sorted(stressAzimuths):
                    avgAperture = sum(frac[azimuth] for frac in fractures if azimuth in frac) / len(fractures)
                    maxAperture = max(frac[azimuth] for frac in fractures if azimuth in frac)
                    minAperture = min(frac[azimuth] for frac in fractures if azimuth in frac)

                    if avgAperture > 1:
                        fileID.write(f"{azimuth} Average: {avgAperture:.3f}\n")
                        fileID.write(f"{azimuth} Max: {maxAperture:.3f}\n")
                        fileID.write(f"{azimuth} Min: {minAperture:.3f}\n")

                    else:
                        fileID.write(f"{azimuth} Average: {avgAperture:.3e}\n")
                        fileID.write(f"{azimuth} Max: {maxAperture:.3e}\n")
                        fileID.write(f"{azimuth} Min: {minAperture:.3e}\n")
                fileID.write("\n")

    def generateOutputFileForOverallProperties(self, name, fractureSets, number=0):

        """
        Generate a text file with overall properties of the fracture network.

        Parameters:
        -----------
        name : str
            Name for the output file.

        fractureSets : list
            List of fracture sets containing fracture data.

        number : int, optional
            Realization number for file naming (default is 0).

        Notes:
        ------
        - Calculates and writes properties such as total number of fractures, intensity, length statistics,
          aperture statistics, connectivity, and proximity to a well.
        - Also computes statistics for corrected apertures based on different stress azimuths.
        - Saves the results in a text file in the specified output directory.
        """
        outputFileForOverallPropertiesDir = os.path.join(self.outputDir, name)
        os.makedirs(outputFileForOverallPropertiesDir, exist_ok=True)
        outputPropertiesFile = os.path.join(outputFileForOverallPropertiesDir, f"{number + 1:03}{name}.txt")

        # Combine all fractures into a single list
        allFractures = [frac for fractures in fractureSets for frac in fractures]
        totalCountFrac = len(allFractures)

        totalIntersection = 0
        # Compute intersections between different sets
        for i, set1 in enumerate(fractureSets):
            for set2 in fractureSets[i + 1:]:
                for frac1 in set1:
                    for frac2 in set2:
                        line1 = ((frac1['x_start'], frac1['y_start']), (frac1['x_end'], frac1['y_end']))
                        line2 = ((frac2['x_start'], frac2['y_start']), (frac2['x_end'], frac2['y_end']))
                        intersect = line_intersection(line1, line2)
                        if intersect:
                            totalIntersection += 1

        areaRock = self.xmax * self.ymax
        connectivity = totalIntersection / (areaRock * totalCountFrac)

        # Compute overall properties
        totalIntensity = self.computeIntensity(allFractures)
        lengths = [frac['fracture length'] for frac in allFractures]
        apertures = [frac['fracture aperture'] for frac in allFractures]

        # Calculate statistics for lengths and apertures
        minLength = min(lengths)
        maxLength = max(lengths)
        avgLength = sum(lengths) / totalCountFrac
        minAperture = min(apertures)
        maxAperture = max(apertures)
        avgAperture = sum(apertures) / totalCountFrac

        wellLocation = (self.xmax / 2, self.ymax / 2)
        minDistanceFromWell = self.distanceOfWellFromClosetFracture(allFractures, wellLocation)

        # Calculate and write correctedAperture values for the current set
        stressAzimuths = set()
        for frac in allFractures:
            for key in frac:
                if key.startswith('correctedAperture'):
                    stressAzimuths.add(key)

        # Write overall properties to the file
        with open(outputPropertiesFile, 'w') as fileID:
            fileID.write(f"Total number of fractures: {totalCountFrac}\n"
                         f"Total intensity: {totalIntensity:.3f}\n"
                         f"Minimum fracture length: {minLength:.3f}\n"
                         f"Maximum fracture length: {maxLength:.3f}\n"
                         f"Average fracture length: {avgLength:.3f}\n"
                         f"Average aperture: {avgAperture:.3e}\n"
                         f"Maximum aperture: {maxAperture:.3e}\n"
                         f"Minimum aperture: {minAperture:.3e}\n"
                         f"connectivity= {connectivity:.3e}\n"
                         f"wellLocation= {wellLocation}\n"
                         f"minDistanceFromWell= {minDistanceFromWell}\n")

            for azimuth in sorted(stressAzimuths):
                avgAperture = sum(frac[azimuth] for frac in allFractures if azimuth in frac) / len(allFractures)
                maxAperture = max(frac[azimuth] for frac in allFractures if azimuth in frac)
                minAperture = min(frac[azimuth] for frac in allFractures if azimuth in frac)
                fileID.write(f"{azimuth} Average: {avgAperture:.3e}\n")
                fileID.write(f"{azimuth} Max: {maxAperture:.3e}\n")
                fileID.write(f"{azimuth} Min: {minAperture:.3e}\n")

    def plotFractures(self, fractureSets, name, number=0):
        """
        Plot and save fractures along with north and stress directions.

        Parameters:
        -----------
        fractureSets : list
            List of fracture sets, each containing fracture data.

        name : str
            Name for the output file.

        number : int, optional
            Realization number for file naming (default is 0).

        Notes:
        ------
        - Fractures from different sets are plotted in distinct colors.
        - Includes a north arrow
        - Saves the plot as a PNG file in the 'pics' directory.
        """
        figDir = os.path.join(self.outputDir, 'pics')
        os.makedirs(figDir, exist_ok=True)
        figDirFile = os.path.join(figDir, f"{number + 1:03}{name}.png")

        ratio = self.ymax / self.xmax
        plt.figure(figsize=(10, 10 * ratio))
        # Colors for different sets of fractures
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']  # Add more colors if needed
        setLabels = [f'Set {i + 1}' for i in range(len(fractureSets))]

        # Plot fractures from each set
        for fractures, color, label in zip(fractureSets, colors, setLabels):
            for fracture in fractures:
                x_start, y_start = fracture['x_start'], fracture['y_start']
                x_end, y_end = fracture['x_end'], fracture['y_end']
                plt.plot([x_start, x_end], [y_start, y_end], color=color, label=label)
                # Use the label only for the first fracture in each set to avoid duplicate legend entries
                label = "_nolegend_"

        # Plot north direction outside the plot
        plt.annotate('N', xy=(1.02, 1.00), xycoords='axes fraction', fontsize=20, ha='center', va='center')
        plt.annotate('', xy=(1.02, 0.98), xytext=(1.02, 0.9), xycoords='axes fraction', textcoords='axes fraction',
                     arrowprops=dict(facecolor='black', shrink=0.05), ha='center', va='center')

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('DFN Visualization')
        plt.grid(True)
        plt.xlim(0, self.xmax)
        plt.ylim(0, self.ymax)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True, shadow=True, ncol=len(fractureSets))
        plt.savefig(figDirFile, bbox_inches='tight', format='png', dpi=300)

    def plotFracturesWithAperture(self, fractureSets, name, number=0):
        """
        Plot and save fractures along with north and stress directions, with thickness and alpha correlated to aperture.

        Parameters:
        -----------
        fractureSets : list
            List of fracture sets, each containing fracture data.

        name : str
            Name for the output file.

        number : int, optional
            Realization number for file naming (default is 0).

        Notes:
        ------
        - Fractures from different sets are plotted in distinct colors.
        - Fracture thickness and transparency are correlated with the fracture aperture.
        - Includes a north arrow.
        - Saves the plot as a PNG file in the 'pics' directory.
        """
        figDir = os.path.join(self.outputDir, 'pics')
        os.makedirs(figDir, exist_ok=True)
        figDirFile = os.path.join(figDir, f"{number + 1:03}{name}.png")

        ratio = self.ymax / self.xmax
        plt.figure(figsize=(10, 10 * ratio))
        # Colors for different sets of fractures
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']  # Add more colors if needed
        setLabels = [f'Set {i + 1}' for i in range(len(fractureSets))]

        max_aperture = max(fracture['fracture aperture'] for fractures in fractureSets for fracture in fractures)
        min_aperture = 0

        # Plot fractures from each set
        for fractures, color, label in zip(fractureSets, colors, setLabels):
            for fracture in fractures:
                x_start, y_start = fracture['x_start'], fracture['y_start']
                x_end, y_end = fracture['x_end'], fracture['y_end']
                aperture = fracture['fracture aperture']

                # Normalize the aperture for thickness and alpha
                norm_aperture = (aperture - min_aperture) / (max_aperture - min_aperture)
                linewidth = 0.5 + 4.5 * norm_aperture  # Line width between 0.5 and 5
                alpha = 0.1 + 0.9 * norm_aperture  # Alpha between 0.2 and 1

                plt.plot([x_start, x_end], [y_start, y_end], color=color,  alpha=alpha, label=label)
                # Use the label only for the first fracture in each set to avoid duplicate legend entries
                label = "_nolegend_"

        # Plot north direction outside the plot
        plt.annotate('N', xy=(1.02, 1.00), xycoords='axes fraction', fontsize=20, ha='center', va='center')
        plt.annotate('', xy=(1.02, 0.98), xytext=(1.02, 0.9), xycoords='axes fraction', textcoords='axes fraction',
                     arrowprops=dict(facecolor='black', shrink=0.05), ha='center', va='center')

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('DFN Visualization with Aperture')
        plt.grid(True)
        plt.xlim(0, self.xmax)
        plt.ylim(0, self.ymax)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True, shadow=True, ncol=len(fractureSets))
        plt.savefig(figDirFile, bbox_inches='tight', format='png', dpi=300)

    def plotCorrectedApertures(self, name, fractureSets, number=0):

        """
        Plot and save fractures colored by corrected apertures for each stress azimuth.

        Parameters:
        -----------
        name : str
            Name for the output file.

        fractureSets : list
            List of fracture sets, where each set contains fractures with corrected aperture data.

        number : int, optional
            Realization number for file naming (default is 0).

        Notes:
        ------
        - Creates subplots for each stress azimuth, coloring fractures based on their corrected apertures.
        - Saves the plot as a PNG file in a directory named 'stressDependancy'.
        - A colorbar is included to indicate the range of corrected aperture values.
        """
        figDir = os.path.join(self.outputDir, 'stressDependancy')
        os.makedirs(figDir, exist_ok=True)
        figDirFile = os.path.join(figDir, f"{number + 1:03}{name}.png")

        # Extract the keys from the first fracture of the first set.
        stress_azimuths = [key for key in fractureSets[0][0].keys() if key.startswith('correctedAperture')]

        num_rows = 2
        num_cols = (len(stress_azimuths) + num_rows - 1) // num_rows  # Ceiling division to handle odd counts
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 18), constrained_layout=True)

        # Determine global min and max corrected aperture values for consistent coloring
        all_apertures = [fracture[key] for set in fractureSets for fracture in set for key in stress_azimuths]
        min_aperture, max_aperture = min(all_apertures), max(all_apertures)

        # Create a colormap
        norm = mcolors.Normalize(vmin=min_aperture, vmax=max_aperture)
        cmap = plt.get_cmap('viridis')


        ax = axes.flat[0]
        ax.set_aspect('equal', adjustable='box')  # Ensure each subplot is square
        for set_index, fractures in enumerate(fractureSets):
            for fracture in fractures:
                corrected_aperture = fracture['initial aperture']
                color = cmap(norm(corrected_aperture))
                x_start, y_start = fracture['x_start'], fracture['y_start']
                x_end, y_end = fracture['x_end'], fracture['y_end']
                ax.plot([x_start, x_end], [y_start, y_end], color=color)

        # Set axis properties
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Initial aperture')
        ax.grid(True)
        ax.set_xlim(0, self.xmax)
        ax.set_ylim(0, self.ymax)


        # Plot fractures for each azimuth in a subplot
        for ax_idx, azimuth in enumerate(stress_azimuths):
            ax = axes.flat[ax_idx+1]
            ax.set_aspect('equal', adjustable='box')  # Ensure each subplot is square
            for set_index, fractures in enumerate(fractureSets):
                for fracture in fractures:
                    corrected_aperture = fracture[azimuth]
                    color = cmap(norm(corrected_aperture))
                    x_start, y_start = fracture['x_start'], fracture['y_start']
                    x_end, y_end = fracture['x_end'], fracture['y_end']
                    ax.plot([x_start, x_end], [y_start, y_end], color=color)

            # Set axis properties
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title(f'{azimuth.replace("correctedAperture", "Stress orientation = ")} ')
            ax.grid(True)
            ax.set_xlim(0, self.xmax)
            ax.set_ylim(0, self.ymax)


        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', pad=0.01)
        cbar.set_label('Corrected Aperture Value', rotation=270, labelpad=15)
        plt.savefig(figDirFile, format='png', dpi=300)
        plt.close(fig)

    def plotOrientationStereographic(self, name, fractureSets, number=0):
        """
        Plot and save a stereographic projection of fracture orientations.

        Parameters:
        -----------
        name : str
            Name for the output directory and file.

        fractureSets : list
            List of fracture sets, where each set contains fractures with orientation data.

        number : int, optional
            Realization number for file naming (default is 0).

        Notes:
        ------
        - Saves the plot as a PNG file in the specified output directory.
        - Each fracture set is plotted with a different color.
        """
        orientationStereographicDir = os.path.join(self.outputDir, name)
        os.makedirs(orientationStereographicDir, exist_ok=True)
        orientationStereographicDirFile = os.path.join(orientationStereographicDir, f"{number + 1:03}{name}.png")
        # Define colors for each set
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Extend this list as needed
        # Plotting
        fig, ax = plt.subplots(subplot_kw={'polar': True})

        for i, set in enumerate(fractureSets):
            theta_values = [frac['theta'] for frac in set]
            theta_radians = np.radians(theta_values)
            ax.hist(theta_radians, bins=36, density=True, alpha=0.75, color=colors[i % len(colors)],
                    label=f'Set {i + 1}')

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        plt.legend()

        plt.savefig(orientationStereographicDirFile, format='png', dpi=300)


def point_to_segment_distance(p, a, b):
    """
    Compute the distance from point p to segment [a, b]

    Parameters:
    -----------
    p : array-like
        Coordinates of the point (x, y).

    a : array-like
        Start of the segment (x, y).

    b : array-like
        End of the segment (x, y).

    Returns:
    --------
    float
        Shortest distance from point `p` to the segment [a, b].

    Notes:
    ------
    - Returns the distance to the nearest endpoint if the point projects outside the segment.
    - If `a` and `b` are the same, returns the distance to that point.
    """

    if np.all(a == b):
        return np.linalg.norm(p - a)
    v = b - a
    w = p - a
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(p - a)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(p - b)

    b = c1 / c2
    pb = a + b * v
    return np.linalg.norm(p - pb)


def segment_to_segment_distance(s1, s2):
    """
    Compute the shortest distance between two line segments s1 and s2.

    Parameters:
    -----------
    s1 : tuple
        First line segment as ((x1, y1), (x2, y2)).

    s2 : tuple
        Second line segment as ((x3, y3), (x4, y4)).

    Returns:
    --------
    float
        Shortest distance between the segments, or 0 if they intersect.
    """

    # first check if they intersect
    intersect, _, _ = line_intersection(s1, s2)
    if intersect:
        return 0
    else:
        s1_start, s1_end = s1
        s2_start, s2_end = s2

        distances = [
            point_to_segment_distance(np.array(s1_start), np.array(s2_start), np.array(s2_end)),
            point_to_segment_distance(np.array(s1_end), np.array(s2_start), np.array(s2_end)),
            point_to_segment_distance(np.array(s2_start), np.array(s1_start), np.array(s1_end)),
            point_to_segment_distance(np.array(s2_end), np.array(s1_start), np.array(s1_end))
        ]
        return min(distances)


def line_intersection(line1, line2):
    """
    Checks if two line segments intersect and returns the intersection point if they do.

    Parameters:
    -----------
    line1 : tuple
        Two points defining the first line segment as ((x1, y1), (x2, y2)).

    line2 : tuple
        Two points defining the second line segment as ((x3, y3), (x4, y4)).

    Returns:
    --------
    tuple
        (bool, float, float):
        - True and the (x, y) coordinates of the intersection if the segments intersect.
        - False and (None, None) if they do not intersect.

    Notes:
    ------
    - Returns False if the lines are parallel.
    - Only returns the intersection point if it lies within both segments.
    """
    ((x1, y1), (x2, y2)) = line1
    ((x3, y3), (x4, y4)) = line2

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    determinant = A1 * B2 - A2 * B1

    if determinant == 0:
        return False, None, None  # Lines are parallel and don't intersect

    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant

    # Check if the intersection point lies within the bounds of both line segments
    if (min(x1, x2) <= x <= max(x1, x2) and
            min(y1, y2) <= y <= max(y1, y2) and
            min(x3, x4) <= x <= max(x3, x4) and
            min(y3, y4) <= y <= max(y3, y4)):
        return True, x, y
    else:
        return False, x, y
