import random
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

from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
class DFNGenerator:

    def __init__(self, domainLengthX, domainLengthY, sets, apertureCalculationParameters, DFNName,bufferZone,IsMultipleStressAzimuths=False, stressAzimuth=None ,savePic=True):
        """
        Parameters:
            -domainLengthX= length of domain in x direction
            -domainLengthY= length of domain in y direction
            - Iy= fracture intensity in vertical direction
            - fractureLengthPDF= choose between: Fixed, Uniform, Log-Normal, Negative power-law, Negative exponential
            - fractureLengthPDFParams a dictionary including different items based on the PDF:
                +"Fixed": fractureLengthPDFParams{"fixed_value": }
                +"Uniform": fractureLengthPDFParams{"Lmin":
                                                    "Lmax":}
                +"Log-Normal" : fractureLengthPDFParams{"sigma":}
                +"Negative power-law": fractureLengthPDFParams{"alpha":
                                                              "Lmin":}
                +"Negative exponential": fractureLengthPDFParams{"lambda":}

        """
        ##  initialization of disturbution functions
        self.xmax=domainLengthX
        self.ymax=domainLengthY
        outputDir = 'DFNs/' + str(DFNName)
        self.outputDir= outputDir
        self.apertureCalculation=apertureCalculator(apertureCalculationParameters,stage='first')
        self.bufferZoneCalculation=bufferZoneCalculator(bufferZone)

        ###############################################################################
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('+++++++ Section A: generate  fractures +++++++++')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


        allFractureSets = []
        for setConfig in sets:
            fractureSet,setConfig  = self.generateFractures(setConfig)
            fractureSet = self.bufferZoneCalculation.calculate(fractureSet)
            fractureSet = self.sortFractures(fractureSet)
            allFractureSets.append((fractureSet, setConfig))

        ###############################################################################
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('+++++++ Section B: placing the fractures +++++++++')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        allProcessedFractureSets = []
        for fractureSet, setConfig in allFractureSets:
            # Now you can use set_config['spatialDisturbutionPDF'] and set_config['spatialDisturbutionPDFParams']
            processedFractureSet = self.place_fractures(fractureSet, setConfig)
            allProcessedFractureSets.append(processedFractureSet)

        ###############################################################################
        # this sub section deals with calculation realted to influence of stress field on aperture
        if IsMultipleStressAzimuths:
            self.stressAzimuth=stressAzimuth
            for azimuth in stressAzimuth:
                apertureCalculationParameters["strike"] = azimuth
                self.apertureCalculation = apertureCalculator(apertureCalculationParameters,stage='second')
                for set in allProcessedFractureSets:
                    correlatedSet = self.apertureCalculation.get_calculator(set)
                    set=correlatedSet
            #self.plotApertureChanges( 'apertureChange',allProcessedFractureSets)
            #self.plotStereographic(allProcessedFractureSets , stressAzimuth,'stratigraphy')


        ###############################################################################
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('+++++++ Section D: Generating the outputs +++++++++')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        if savePic:
            self.plotFractures(allProcessedFractureSets,name= 'DFNPic')
        self.generateOutputFileForOutputPropertiesPerSet('outputPropertiesPerSet', allProcessedFractureSets)
        self.generateOutputFileForOverallProperties('outputPropertiesTotal', allProcessedFractureSets)
        self.generateTextFileForFractureCoordinates( 'fractureCoordinates',allProcessedFractureSets)
        self.generateTextFileForFractureApertures('aperture', allProcessedFractureSets)
        self.generateInputPropertiesFile( 'inputProperties', sets, stressAzimuth)
        self.plotOrientationStereographic( 'orientationStereographic',allProcessedFractureSets)
        if IsMultipleStressAzimuths:
            self.generateTextFilesForCorrectedApertures('correlatedAperture', allProcessedFractureSets)
            self.plotCorrectedApertures('aperturePerStrikeTotal', allProcessedFractureSets)
            self.plotCorrectedAperturesJutForPaper('aperturePerStrike', allProcessedFractureSets)
            #self.plotStereographicWithRose('roseDiagram', allProcessedFractureSets)





    def placeLongestFracture(self,longestFractrue):
        # Seed Selection and placing the longest fracture
        referenceWithinDomain= False
        number=0
        tries = 0
        print('fracture length',longestFractrue['fracture length'])
        while not referenceWithinDomain:
            seed_x = random.uniform(0, self.xmax)
            seed_y = random.uniform(0, self.ymax)
            (new_x_start, new_y_start), (new_x_end, new_y_end) = self.fractureCordinate(longestFractrue,90-longestFractrue['theta'], seed_x,seed_y)
            if self.is_within_domain(new_x_start, new_y_start) and self.is_within_domain(new_x_end, new_y_end):
                referenceWithinDomain = True
                addedFracture = {
                    'number': number,
                    'x_start': new_x_start,
                    'y_start': new_y_start,
                    'x_end': new_x_end,
                    'y_end': new_y_end,
                    'fracture length': longestFractrue['fracture length'],
                    'fracture spacing': longestFractrue['fracture spacing'],
                    'theta': longestFractrue['theta'],
                }
            else:
                tries=+1
                if tries>500:
                    return False

        return (addedFracture,seed_x,seed_y)

    def place_fractures(self,fractures, setConfig):

        spatialDisturbutionPDF = spatialDisturbutionPDFs(setConfig['spatialDisturbutionPDF'], setConfig['spatialDisturbutionPDFParams'])
        spatialDisturbutionPDFMode = spatialDisturbutionPDF.compute_mode()


        # Step A2: Fracture Placement
        processedFractures=[]
        print('--- placing the longest fracture----')
        starting=0
        added=False
        while not added:
            added =self.placeLongestFracture(fractures[0])

        (addedFracture, seed_x, seed_y)=added

        processedFractures.append(addedFracture)
        print('--the longest fracture added--')
        numberOfTries=0
        number = 0
        # Introducing New Fractures
        print('--- placing rest of fractures ---')
        for fracture in fractures[1:]:
            theta = 90-fracture['theta']
            isNewFractureAdded=False
            while not isNewFractureAdded:
                referenceWithinDomain = False
                while not referenceWithinDomain:
                    distance =  (spatialDisturbutionPDF.get_value() - spatialDisturbutionPDFMode )* random.choice([-1, 1]) # Randomness for + and -
                    if theta<90:
                        new_x_mid = seed_x + np.random.randint(0, self.ymax) * random.choice([-1, 1])
                        new_y_mid = seed_y + distance
                    else:
                        new_x_mid = seed_x + distance
                        new_y_mid = seed_y + np.random.randint(0, self.ymax)* random.choice([-1, 1])
                    (new_x_start, new_y_start), (new_x_end, new_y_end) = self.fractureCordinate(fracture, theta, new_x_mid,new_y_mid)
                    #add check proximity here
                    if self.is_within_domain(new_x_start, new_y_start) and self.is_within_domain(new_x_end, new_y_end):
                        referenceWithinDomain = True
                # Check proximity
                too_close = False
                for existing_frac in processedFractures:
                    existing_coords = ((existing_frac['x_start'], existing_frac['y_start']),
                                       (existing_frac['x_end'], existing_frac['y_end']))
                    new_coords = ((new_x_start, new_y_start), (new_x_end, new_y_end))
                    if segment_to_segment_distance(new_coords, existing_coords) < existing_frac['fracture spacing']+fracture['fracture spacing']:
                        too_close = True
                        numberOfTries+=1
                        print( "fracture number",str(fracture['number']),'with length of ', str(fracture['fracture length']),' has been relocated. I= ', str(self.computeIntensity(processedFractures)),"has been reached",str(numberOfTries), "times")
                        break
                if not too_close:
                    isNewFractureAdded= True
                    number += 1
                    addedFracture = {
                        'number': number,
                        'x_start': new_x_start,
                        'y_start': new_y_start,
                        'x_end': new_x_end,
                        'y_end': new_y_end,
                        'fracture length': fracture['fracture length'],
                        'fracture spacing': fracture['fracture spacing'],
                        'theta': fracture['theta']
                    }
                    processedFractures.append(addedFracture)
            if numberOfTries > 100000:
                print("Max retries reached for vertical fractures, fracture generation for y stopped. Intensity is= ",
                      str(self.computeIntensity(processedFractures)))
                break

        #this i when we add fractures aperture
        processedFractures=self.apertureCalculation.get_calculator(processedFractures)
        #self.computeAperture_SubLinearLengthApertureScaling(processedFractures)



        return processedFractures

    def generateFractures(self,setConfig):

        """
        Parameters:
        - fractureIntensity (float): Desired fracture intensity for the network.

        Returns:
        -a list of dictionaries with fractures properties sorted by fracture length and each dictionary including 'fracture length'
        """
        if setConfig['orientationDisturbutonPDF']=='Constant':
            theta= setConfig['orientationDisturbutonPDFParams']['theta']
        elif setConfig['orientationDisturbutonPDF']=='Von-Mises':
            theta= np.degrees(setConfig['orientationDisturbutonPDFParams']["loc"])
        elif setConfig['orientationDisturbutonPDF']=='Uniform':
            theta= (setConfig['orientationDisturbutonPDFParams']["max theta"]+setConfig['orientationDisturbutonPDFParams']["min theta"])/2


        thetaYforGeneration = 90 - theta
        if thetaYforGeneration>90:
            thetaYforGeneration=180-thetaYforGeneration

        if thetaYforGeneration >= 45:
            thetaYforGeneration=90-thetaYforGeneration
            thetaRadian = np.radians(thetaYforGeneration)
            setConfig['fractureLengthPDFParams']["Lmax"]= self.ymax / math.cos(thetaRadian)
        else:
            thetaRadian = np.radians(thetaYforGeneration)
            setConfig['fractureLengthPDFParams']["Lmax"]= self.xmax / math.cos(thetaRadian)

        #fractureIntensity,theta,nameFractureLengthPDF,fractureLengthPDFParams):
        orientationDisturbutonPDF=orientationPDFs(setConfig['orientationDisturbutonPDF'], setConfig['orientationDisturbutonPDFParams'])
        theta=orientationDisturbutonPDF.get_value()


        print('fractureLengthPDFParams["Lmax"]=',setConfig['fractureLengthPDFParams']["Lmax"])

        fractureLengthPDF = fractureLengthPDFs(setConfig['fractureLengthPDF'], setConfig['fractureLengthPDFParams'])
        fractures=[]
        newFrac={}
        newFrac['fracture length']=fractureLengthPDF.get_value()
        newFrac['theta'] = theta
        fractures.append(newFrac)
        while self.computeIntensity(fractures) < setConfig['I']:
            newFrac = {}
            newFrac['fracture length'] = fractureLengthPDF.get_value()
            newFrac['theta']=orientationDisturbutonPDF.get_value()
            fractures.append(newFrac)

        return fractures, setConfig

    def computeIntensity(self, fractures):
        total_length = sum([fracture['fracture length'] for fracture in fractures])
        area = self.xmax * self.ymax
        intensity = total_length / area
        return intensity

    def sortFractures(self, fractures):
        # Sort the fractures based on 'fracture length' in descending order
        fractures.sort(key=lambda x: x['fracture length'], reverse=True)
        # Assign a 'number' to each fracture starting from 0 for the longest
        for i, fracture in enumerate(fractures, start=0):
            fracture['number'] = i
        return fractures

    def fractureCordinate(self,fracture,theta, midX, midY):
        half_length = fracture['fracture length'] / 2  # the longest fracture
        new_x_start = midX - half_length * np.cos(np.radians(theta))
        new_y_start = midY - half_length * np.sin(np.radians(theta))
        new_x_end = midX + half_length * np.cos(np.radians(theta))
        new_y_end = midY + half_length * np.sin(np.radians(theta))
        return (new_x_start,new_y_start), (new_x_end,new_y_end)

    def is_within_domain(self, x, y):
        return 0 <= x <= self.xmax and 0 <= y <= self.ymax

    def distanceOfWellFromClosetFracture(self, fractures, wellLocation):
        min_distance = float('inf')
        for frac in fractures:
            x_start, y_start = frac['x_start'], frac['y_start']
            x_end, y_end = frac['x_end'], frac['y_end']
            distance = point_to_segment_distance(np.array(wellLocation), np.array([x_start, y_start]),
                                                 np.array([x_end, y_end]))
            min_distance = min(min_distance, distance)
        return min_distance

    #### outputs
    def generateInputPropertiesFile(self,name, fractureSets,stressAzimuth,  number=0):
        """
        This function generates the input properties file.
        """
        InputPropertiesDir = self.outputDir + '/'+str(name)
        os.makedirs(InputPropertiesDir, exist_ok=True)
        InputPropertiesFile = f"{InputPropertiesDir}\\{name}{number + 1:03}.txt"
        with open(InputPropertiesFile, 'w') as f:
            f.write("domainLengthX: " + str(self.xmax) + '\n')
            f.write("domainLengthY: " + str(self.ymax) + '\n')

            i=1
            for set in fractureSets:
                if isinstance(set, dict):
                    # Convert dictionary to a pretty-printed JSON string for readability
                    import json
                    f.write(
                        "fractureSet "+str(i)+" : " + json.dumps(set, indent=4) + '\n')
                else:
                    f.write("fractureSet " + str(i)+" : " + '\n')
                i=+1


            f.write("stressAzimuth: " + str(stressAzimuth) + '\n')

    def generateTextFileForFractureCoordinates(self, name,fractures,number=0):
        """
        This function generates the input properties file.
        """
        coordinatesOutputDir = self.outputDir + '/'+str(name)
        os.makedirs(coordinatesOutputDir, exist_ok=True)
        InputPropertiesFile = f"{coordinatesOutputDir}\\{name}{number + 1:03}.txt"
        with open(InputPropertiesFile, 'w') as fileID:
            for set in fractures:
                for frac in set:
                    fileID.write(f"{frac['x_start']:.4f} {frac['y_start']:.4f} {frac['x_end']:.4f} {frac['y_end']:.4f}\n")

    def generateTextFileForFractureApertures(self, name, fractures,number=0):
        """
        This function generates the input properties file.
        """
        apertureOutputDir = self.outputDir + '/'+str(name)
        os.makedirs(apertureOutputDir, exist_ok=True)
        InputPropertiesFile = f"{apertureOutputDir}\\{name}{number + 1:03}.txt"

        with open(InputPropertiesFile, 'w') as fileID:
            for set in fractures:
                for frac in set:
                    fileID.write(f"{frac['fracture aperture']:.7f}\n")

    def generateTextFilesForCorrectedApertures(self, name, fractures, number=0):
        """
        This function generates separate text files for each corrected aperture.
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
        outputFileForOutputPropertiesDir = self.outputDir + '/' + str(name)
        os.makedirs(outputFileForOutputPropertiesDir, exist_ok=True)
        outputPropertiesFile = f"{outputFileForOutputPropertiesDir}/{name}{number + 1:03}.txt"
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

        outputFileForOverallPropertiesDir = self.outputDir + '/' + str(name)
        os.makedirs(outputFileForOverallPropertiesDir, exist_ok=True)
        outputPropertiesFile = f"{outputFileForOverallPropertiesDir}/{name}{number + 1:03}.txt"

        # Combine all fractures into a single list
        allFractures = [frac for fractures in fractureSets for frac in fractures]
        totalCountFrac = len(allFractures)

        totalIntersection=0
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

        areaRock=self.xmax*self.ymax
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
    """
    def plotApertureChanges(self, name, fractureSets, number=0):
        figDir = self.outputDir + '/stressDependancy'
        os.makedirs(figDir, exist_ok=True)
        figDirFile = f"{figDir}\\{name}{number + 1:03}.png"

        # Dictionary to store percent differences for each fracture set
        stress_azimuths = [key for key in fractureSets[0][0].keys() if key.startswith('correctedAperture')]

        thetaValues = [np.mean([frac['theta'] for frac in fractureSet]) for fractureSet in fractureSets]
        percent_diffs = {theta: [] for theta in thetaValues}

        for azimuth in stress_azimuths:
            for theta, fractures in zip(thetaValues, fractureSets):
                # Retrieve the average aperture before deformation
                avg_aperture_before = np.mean([frac['initial aperture'] for frac in fractures])
                # Retrieve the average aperture after deformation for the current azimuth
                avg_aperture_after = np.mean([frac[str(azimuth)] for frac in fractures])
                # Calculate the percent differences and store
                percent_diff = 100 * (avg_aperture_before - avg_aperture_after) / avg_aperture_before
                percent_diffs[theta].append(percent_diff)

        # Plotting
        fig, ax = plt.subplots()

        # Generate bar positions for each azimuth and each set of fractures
        num_azimuths = len(stress_azimuths)
        num_sets = len(fractureSets)
        width = 0.8 / num_sets
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # Add more colors if needed

        for i, (theta, diffs) in enumerate(percent_diffs.items()):
            bar_positions = np.arange(num_azimuths) * num_sets - i * width
            ax.bar(bar_positions, diffs, width=width, label=f'orientation= {theta}', color=colors[i % len(colors)])

        # Setting the title, labels, and legend
        ax.set_title('Deformed Fracture Aperture (%)')
        ax.set_xticks(np.arange(num_azimuths) * num_sets-0.5)
        azimuth_number = [float(azimuth.replace("correctedAperture", "")) for azimuth in stress_azimuths ]
        ax.set_xticklabels(['azimuth=' + str(a) for a in azimuth_number])
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(figDirFile, format='png', dpi=300)
    
    def plotStereographic(self,fractureSets, shmax_azimuths,name,number=0):
        figDir = self.outputDir + '/stressDependancy'
        os.makedirs(figDir, exist_ok=True)
        figDirFile = f"{figDir}\\{name}{number + 1:03}.png"

        thetaValues = [np.mean([frac['theta'] for frac in fractureSet]) for fractureSet in fractureSets]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        # Plotting the circle
        circle = plt.Circle((0, 0), 1, color='green', fill=False)
        ax.add_artist(circle)
        num_sets = len(thetaValues)

        fracture_labels= ['Set '+str(i+1) for i in np.arange(num_sets)]



        # Plotting and labeling fractures
        for orientation, label in zip(thetaValues, fracture_labels):
            x = [np.sin(np.radians(orientation)), -np.sin(np.radians(orientation))]
            y = [np.cos(np.radians(orientation)), -np.cos(np.radians(orientation))]
            ax.plot(x, y)
            ax.text(x[1] * 1.1, y[1] * 1.1, label, ha="center", va="center")

        # Plotting the S_hmax azimuths
        fixed_arrow_length = 0.2  # Adjust this to your desired arrow length
        distance_from_center = 1.2  # Adjust this to change arrow's starting distance from circle center
        # Plotting the S_hmax azimuths
        cmap = get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=360)
        for azimuth in shmax_azimuths:
            color = cmap(norm(azimuth))
            x = np.sin(np.radians(azimuth))
            y = np.cos(np.radians(azimuth))

            # Starting points for arrow
            start_x = x * distance_from_center
            start_y = y * distance_from_center

            # Calculate the x and y components of the arrow length based on the azimuth
            delta_x = fixed_arrow_length * np.sin(np.radians(azimuth))
            delta_y = fixed_arrow_length * np.cos(np.radians(azimuth))

            # Ending points for arrow
            end_x = start_x + delta_x
            end_y = start_y + delta_y

            # textPosition
            xText = start_x + delta_x*2
            yText = start_y + delta_y*2

            ax.arrow(start_x, start_y, delta_x, delta_y, head_width=0.05, head_length=0.1, fc=color, ec=color)
            ax.text(xText, yText, str(azimuth), ha="center", va="center")
            x = np.sin(np.radians(azimuth+180))
            y = np.cos(np.radians(azimuth+180))

            # Starting points for arrow
            start_x = x * distance_from_center
            start_y = y * distance_from_center

            # Calculate the x and y components of the arrow length based on the azimuth
            delta_x = fixed_arrow_length * np.sin(np.radians(azimuth+180))
            delta_y = fixed_arrow_length * np.cos(np.radians(azimuth+180))

            # Ending points for arrow
            end_x = start_x + delta_x
            end_y = start_y + delta_y
            ax.arrow(start_x, start_y, delta_x, delta_y, head_width=0.05, head_length=0.1, fc=color, ec=color)

        ax.set_aspect('equal', 'box')
        ax.axis('off')
        plt.savefig(figDirFile, format='png', dpi=300)
    """

    def plotFractures(self, fractureSets,  name, number=0):
        """Plot the fractures, north direction, and stress directions and save the output to a PDF."""

        figDir = self.outputDir + '/pics'
        os.makedirs(figDir, exist_ok=True)
        figDirFile = f"{figDir}\\{name}{number + 1:03}.png"

        plt.figure(figsize=(12, 12))

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
        #plt.annotate('', xy=(1.02, 1.02), xycoords='axes fraction', fontsize=20, ha='center', va='center')
        plt.annotate('', xy=(1.02, 0.98), xytext=(1.02, 0.9), xycoords='axes fraction', textcoords='axes fraction',
                     arrowprops=dict(facecolor='black', shrink=0.05), ha='center', va='center')

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('DFN Visualization')
        plt.grid(True)
        plt.xlim(0, self.xmax)
        plt.ylim(0, self.ymax)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True, shadow=True, ncol=len(fractureSets))

        # Save the plot to a PNG
        plt.savefig(figDirFile, format='png', dpi=300)

    def plotCorrectedApertures(self, name, fractureSets, number=0):

        """Plot the fractures with colors based on their corrected apertures for each stress azimuth."""
        figDir = self.outputDir + '/stressDependancy'
        os.makedirs(figDir, exist_ok=True)
        figDirFile = f"{figDir}\\{name}{number + 1:03}.png"

        # Extract the keys from the first fracture of the first set.
        stress_azimuths = [key for key in fractureSets[0][0].keys() if key.startswith('correctedAperture')]

        # Number of rows and columns for subplots
        num_rows = 2
        num_cols = (len(stress_azimuths) + num_rows - 1) // num_rows  # Ceiling division to handle odd counts

        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*6, num_rows*6), constrained_layout=True)

        # Determine global min and max corrected aperture values for consistent coloring
        all_apertures = [fracture[key] for set in fractureSets for fracture in set for key in stress_azimuths]
        min_aperture, max_aperture = min(all_apertures), max(all_apertures)

        # Create a colormap
        norm = mcolors.Normalize(vmin=min_aperture, vmax=max_aperture)
        cmap = plt.get_cmap('viridis')

        # Plot fractures for each azimuth in a subplot
        for ax_idx, azimuth in enumerate(stress_azimuths):
            ax = axes.flat[ax_idx]
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
            ax.set_title(f'{azimuth.replace("correctedAperture", "Azimuth ")} Visualization')
            ax.grid(True)
            ax.set_xlim(0, self.xmax)
            ax.set_ylim(0, self.ymax)

        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # New colorbar code
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', pad=0.01)
        cbar.set_label('Corrected Aperture Value', rotation=270, labelpad=15)
        plt.savefig(figDirFile, format='png', dpi=300)
        # Save the figure
        #figPath = os.path.join(self.outputDir, f"{name}_corrected_aperture{number + 1:03}.png")
        #plt.savefig(figPath, format='png', dpi=300)
        plt.close(fig)

    def plotOrientationStereographic(self, name, fractureSets, number=0):
        orientationStereographicDir = self.outputDir + '/' + str(name)
        os.makedirs(orientationStereographicDir, exist_ok=True)
        orientationStereographicDirFile = f"{orientationStereographicDir}\\{name}{number + 1:03}.png"
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

    def plotCorrectedAperturesJutForPaper(self, name, fractureSets, number=0):
        figDir = self.outputDir + '/stressDependancy'
        os.makedirs(figDir, exist_ok=True)
        figDirFile = f"{figDir}\\{name}{number + 1:03}.png"
        """Plot the fractures with colors based on their corrected apertures for each stress azimuth."""

        # Extract the keys from the first fracture of the first set.
        stress_azimuths = [key for key in fractureSets[0][0].keys() if key.startswith('correctedAperture')]

        # Number of rows and columns for subplots
        num_rows = 1
        num_cols = 3 # Ceiling division to handle odd counts

        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*6, num_rows*6), constrained_layout=True)

        # Determine global min and max corrected aperture values for consistent coloring
        all_apertures = [fracture[key] for set in fractureSets for fracture in set for key in stress_azimuths]
        min_aperture, max_aperture = min(all_apertures), max(all_apertures)

        # Create a colormap
        norm = mcolors.Normalize(vmin=min_aperture, vmax=max_aperture)
        cmap = plt.get_cmap('viridis')

        # Plot fractures for each azimuth in a subplot
        for ax_idx, azimuth in enumerate(stress_azimuths):
            ax = axes.flat[ax_idx]
            ax.set_aspect('equal', adjustable='box')  # Ensure each subplot is square
            for set_index, fractures in enumerate(fractureSets):
                for fracture in fractures:
                    corrected_aperture = fracture[azimuth]
                    color = cmap(norm(corrected_aperture))
                    x_start, y_start = fracture['x_start'], fracture['y_start']
                    x_end, y_end = fracture['x_end'], fracture['y_end']
                    ax.plot([x_start, x_end], [y_start, y_end], color=color)
            # Plot north direction outside the plot
            ax.annotate('N', xy=(1.02, 1.00), xycoords='axes fraction', fontsize=15, ha='center', va='center')
            # plt.annotate('', xy=(1.02, 1.02), xycoords='axes fraction', fontsize=20, ha='center', va='center')
            ax.annotate('', xy=(1.02, 0.98), xytext=(1.02, 0.9), xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(facecolor='black', width=2), ha='center', va='center')
            # Extract the azimuth number from the variable, assuming it follows 'correctedAperture'
            azimuth_number = float(azimuth.replace("correctedAperture", ""))
            ax.set_title(r'$\sigma_{{\mathrm{{Hmax}}}}$ strike = {}'.format(azimuth_number),fontsize=15)

            # Add the strike direction arrow next to each subplot title
            # The arrow will be a simple line for simplicity, pointing in the direction of the azimuth
            # Arrow parameters and position need to be adjusted to properly fit the subplot
            arrow_length = 0.1  # Length of the arrow (adjust as needed)
            angle = np.radians(azimuth_number)  # Convert azimuth number to radians
            dx = arrow_length * np.cos(angle)  # Change in x based on the angle
            dy = arrow_length * np.sin(angle)  # Change in y based on the angle
            ax.annotate('', xy=(0.8 + dx, 1.02 + dy), xytext=(0.8, 1.02), xycoords='axes fraction',
                        textcoords='axes fraction', arrowprops=dict(facecolor='black', width=2))

            # Set axis properties
            ax.set_xlabel('X Coordinate',fontsize=12)
            ax.set_ylabel('Y Coordinate',fontsize=12)
            # Extract the azimuth number from the variable, assuming it follows 'correctedAperture'
            azimuth_number = azimuth.replace("correctedAperture", "")
            ax.set_title(r'$\sigma_{{\mathrm{{Hmax}}}}$ strike = {}'.format(azimuth_number), size=15)  # Title
            # Then use this number in the title
            #ax.set_title(r'$\sigma_{{\mathrm{{Hmax}}}}$ strike = {}'.format(azimuth_number))

            ax.grid(True)
            ax.set_xlim(0, self.xmax)
            ax.set_ylim(0, self.ymax)

        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # New colorbar code
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', pad=0.05)
        cbar.set_label('Corrected Aperture Value', rotation=270, labelpad=15,fontsize=15)

        # Save the figure
        plt.savefig(figDirFile, format='png', dpi=300)
        plt.close(fig)



def point_to_segment_distance( p, a, b):
    """Compute the distance from point p to segment [a, b]."""
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
    """Compute the distance between two segments s1 and s2."""
    s1_start, s1_end = s1
    s2_start, s2_end = s2

    distances = [
        point_to_segment_distance(np.array(s1_start), np.array(s2_start), np.array(s2_end)),
        point_to_segment_distance(np.array(s1_end), np.array(s2_start), np.array(s2_end)),
        point_to_segment_distance(np.array(s2_start), np.array(s1_start), np.array(s1_end)),
        point_to_segment_distance(np.array(s2_end), np.array(s1_start), np.array(s1_end))
    ]
    return min(distances)

def is_point_on_line_segment(point, line1):
    (x, y) = point
    ((x1, y1), (x2, y2)) = line1

    # Check if point is out of bounds
    if (x < min(x1, x2)) or (x > max(x1, x2)) or (y < min(y1, y2)) or (y > max(y1, y2)):
        return False

    # Handle vertical line case
    if x1 == x2:
        return x == x1

    # Calculate slope of line segment
    m = (y2 - y1) / (x2 - x1)

    # Check if y-coordinate of point matches the line equation
    return abs(y - (m * (x - x1) + y1)) < 1e-9
def line_intersection(line1, line2):
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
# Function to calculate distance between two points
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5
def point_to_line_distance(point, line):
    x0, y0 = point
    x1, y1, x2, y2 = line

    # If the line segment is vertical
    if x1 == x2:
        # If the point's y-coordinate is between the y-coordinates of the line segment's endpoints
        if min(y1, y2) <= y0 <= max(y1, y2):
            distance = abs(x0 - x1)
        else:
            dist_to_start = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            dist_to_end = math.sqrt((x0 - x2)**2 + (y0 - y2)**2)
            distance = min(dist_to_start, dist_to_end)
        slope = "undefined"
    else:
        # Calculate the slope of the line
        m = (y2 - y1) / (x2 - x1)
        # Convert the line to the form ax + by + c = 0
        a = m
        b = -1
        c = y1 - m * x1
        # Calculate the perpendicular distance
        distance = abs(a * x0 + b * y0 + c) / math.sqrt(a**2 + b**2)
        slope = m

        # Check if the perpendicular from the point to the line falls outside the segment
        dot1 = (x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)
        dot2 = (x0 - x2) * (x1 - x2) + (y0 - y2) * (y1 - y2)
        if dot1 * dot2 > 0:
            dist_to_start = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            dist_to_end = math.sqrt((x0 - x2)**2 + (y0 - y2)**2)
            distance = min(dist_to_start, dist_to_end)

    return distance




def merge_fractures(fracX, fracY, PrevStartAd, prevEndAd):
    isAdjustedFracYStart, isAdjustedFracYEnd = PrevStartAd, prevEndAd
    line1 = ((fracX['x_start'], fracX['y_start']), (fracX['x_end'], fracX['y_end']))
    line2 = ((fracY['x_start'], fracY['y_start']), (fracY['x_end'], fracY['y_end']))

    intersect, x, y = line_intersection(line1, line2)

    if intersect:
        return fracX, fracY, isAdjustedFracYStart, isAdjustedFracYEnd
    line1 = (np.array([fracX['x_start'], fracX['y_start']]), np.array([fracX['x_end'], fracX['y_end']]))
    line2 = (np.array([fracY['x_start'], fracY['y_start']]), np.array([fracY['x_end'], fracY['y_end']]))

    # Find the longer fracture
    lenX = fracX['fracture length']
    lenY = fracY['fracture length']
    threshold=5 * (fracX['fracture spacing'] if lenX > lenY else fracY['fracture spacing']*5)
    if segment_to_segment_distance(line1, line2) > threshold:
        return fracX, fracY, isAdjustedFracYStart, isAdjustedFracYEnd

    def adjust_point(frac, point_key, x, y):
        frac['x_'+point_key], frac['y_'+point_key] = x, y

    # Scenario A: The intersection point lies on fracX
    if is_point_on_line_segment((x, y), line1):
        dist_start = point_to_segment_distance(line2[0], line1[0], line1[1])
        dist_end = point_to_segment_distance(line2[1], line1[0], line1[1])
        if dist_start< threshold or dist_end<threshold:
            if dist_start < dist_end:
                if not isAdjustedFracYStart:
                    adjust_point(fracY, 'start', x, y)
                    isAdjustedFracYStart = True
            else:
                if not isAdjustedFracYEnd:
                    adjust_point(fracY, 'end', x, y)
                    isAdjustedFracYEnd = True

    # Scenario B: The intersection point lies on fracY
    elif is_point_on_line_segment((x, y), line2):
        dist_start = point_to_segment_distance(line1[0], line2[0], line2[1])
        dist_end = point_to_segment_distance(line1[1], line2[0], line2[1])
        if dist_start < threshold or dist_end < threshold:
            if dist_start < dist_end:
                adjust_point(fracX, 'start', x, y)
            else:
                adjust_point(fracX, 'end', x, y)

    # Scenario C: The intersection point doesn’t lie on fracX or fracY
    else:
        # Adjust endpoint of fracY closest to the intersection point
        dist_start = point_to_segment_distance(line2[0], line1[0], line1[1])
        dist_end = point_to_segment_distance(line2[1], line1[0], line1[1])
        if dist_start < threshold or dist_end < threshold:
            if dist_start < dist_end:
                if not isAdjustedFracYStart:
                    adjust_point(fracY, 'start', x, y)
                    isAdjustedFracYStart = True
            else:
                if not isAdjustedFracYEnd:
                    adjust_point(fracY, 'end', x, y)
                    isAdjustedFracYEnd = True

        # Adjust endpoint of fracX closest to the intersection point
        dist_start = point_to_segment_distance(line1[0], line2[0], line2[1])
        dist_end = point_to_segment_distance(line1[1], line2[0], line2[1])
        if dist_start < threshold or dist_end < threshold:
            if dist_start < dist_end:
                adjust_point(fracX, 'start', x, y)
            else:
                adjust_point(fracX, 'end', x, y)
    return fracX, fracY, isAdjustedFracYStart, isAdjustedFracYEnd