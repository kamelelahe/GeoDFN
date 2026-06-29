import logging
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from .apertureCalculator import apertureCalculator
from .bufferZoneCalculator import bufferZoneCalculator
from .fractureLengthPDFs import fractureLengthPDFs
from .spatialDistributionPDFs import SpatialDistributionPDFs
from .orientationPDFs import OrientationPDFs
from ._validation import validate_inputs

import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)


class DFNGenerator:

    def __init__(self, domainLengthX, domainLengthY, sets, apertureCalculationParameters, DFNName,
                 numOfRealizations=1, IsMultipleStressAzimuths=False, stressAzimuth=None, savePic=True,
                 output_dir='DFNs', progress_callback=None):
        validate_inputs(domainLengthX, domainLengthY, sets, apertureCalculationParameters, numOfRealizations)
        self.maxtries = []
        self.xmax = domainLengthX
        self.ymax = domainLengthY
        self.outputDir = os.path.join(output_dir, str(DFNName))
        self.apertureCalculation = apertureCalculator(apertureCalculationParameters, stage='first')
        self.numberOfMaxTries = 400000
        self._sets = sets
        self._apertureCalculationParameters = apertureCalculationParameters
        self._numOfRealizations = numOfRealizations
        self._IsMultipleStressAzimuths = IsMultipleStressAzimuths
        self._stressAzimuth = stressAzimuth
        self._savePic = savePic
        self._progress_callback = progress_callback
        self.generate()

    def generate(self):
        sets = self._sets
        apertureCalculationParameters = self._apertureCalculationParameters
        IsMultipleStressAzimuths = self._IsMultipleStressAzimuths
        stressAzimuth = self._stressAzimuth
        savePic = self._savePic
        self.realizations = []

        for i in range(self._numOfRealizations):
            logger.info('Section A: generate fractures')
            self.maxtries = []
            allFractureSets = []
            self.setNumber = 1
            for setConfig in sets:
                fractureSet, setConfig = self._generate_fractures(setConfig)
                fractureSet = self._sort_fractures(fractureSet)
                allFractureSets.append((fractureSet, setConfig))
                self.setNumber = self.setNumber + 1

            logger.info('Section B: placing the fractures')
            allProcessedFractureSets = []
            for fractureSet, setConfig in allFractureSets:
                logger.debug('bufferZone: %s', setConfig['bufferZone'])
                self.bufferZoneCalculation = bufferZoneCalculator(setConfig['bufferZone'])
                fractureSet = self.bufferZoneCalculation.calculate(fractureSet)
                processedFractureSet = self.place_fractures(fractureSet, setConfig)
                allProcessedFractureSets.append(processedFractureSet)

            logger.info('maxtries= %s', self.maxtries)

            maxtriesDir = os.path.join(self.outputDir, 'tries')
            os.makedirs(maxtriesDir, exist_ok=True)
            maxtriesFile = os.path.join(maxtriesDir, 'tries.txt')
            with open(maxtriesFile, 'w') as fileID:
                fileID.write(f"Number of iterations for each set: {self.maxtries}\n")

            if not any(t > self.numberOfMaxTries for t in self.maxtries):
                self.realizations.append(allProcessedFractureSets)
                if IsMultipleStressAzimuths:
                    self._stressAzimuth = stressAzimuth
                    for azimuth in stressAzimuth:
                        apertureCalculationParameters["strike"] = azimuth
                        self.apertureCalculation = apertureCalculator(apertureCalculationParameters, stage='second')
                        for fracture_set in allProcessedFractureSets:
                            fracture_set = self.apertureCalculation.get_calculator(fracture_set)

                logger.info('Section D: Generating the outputs')

                if savePic:
                    self._plot_fractures(allProcessedFractureSets, name='DFNPic', number=i)

                self._write_output_properties_per_set('outputPropertiesPerSet', allProcessedFractureSets, number=i)
                self._write_overall_properties('outputPropertiesTotal', allProcessedFractureSets, number=i)
                self._write_fracture_coordinates('fractureCoordinates', allProcessedFractureSets, number=i)
                self._write_fracture_set('fractureSet', allProcessedFractureSets, number=i)
                self._write_fracture_apertures('aperture', allProcessedFractureSets, number=i)
                self._write_input_properties('inputProperties', apertureCalculationParameters, sets, stressAzimuth, number=i)
                self._plot_orientation_stereographic('orientationStereographic', allProcessedFractureSets, number=i)
                if IsMultipleStressAzimuths:
                    self._write_corrected_apertures('correlatedAperture', allProcessedFractureSets, number=i)
                    self._plot_corrected_apertures('aperturePerStrikeTotal', allProcessedFractureSets, number=i)

            if self._progress_callback:
                self._progress_callback(i + 1, self._numOfRealizations)

    def _place_longest_fracture(self, longestFracture):
        referenceWithinDomain = False
        number = 0
        tries = 0
        logger.debug('fracture length %s', longestFracture['fracture length'])
        while not referenceWithinDomain:
            seed_x = random.uniform(0, self.xmax)
            seed_y = random.uniform(0, self.ymax)
            (new_x_start, new_y_start), (new_x_end, new_y_end) = self._fracture_coordinate(
                longestFracture, longestFracture['theta'], seed_x, seed_y)
            if self._is_within_domain(new_x_start, new_y_start) and self._is_within_domain(new_x_end, new_y_end):
                referenceWithinDomain = True
                addedFracture = {
                    'number': number,
                    'x_start': new_x_start,
                    'y_start': new_y_start,
                    'x_end': new_x_end,
                    'y_end': new_y_end,
                    'fracture length': longestFracture['fracture length'],
                    'set number': longestFracture['set number'],
                    'fracture spacing': longestFracture['fracture spacing'],
                    'theta': longestFracture['theta'],
                }
            else:
                tries += 1
                if tries > 500:
                    return False
        return (addedFracture, seed_x, seed_y)

    # Keep old name as alias for backward compatibility
    def placeLongestFracture(self, longestFracture):
        return self._place_longest_fracture(longestFracture)

    def place_fractures(self, fractures, setConfig):
        spatialDistributionPDF = SpatialDistributionPDFs(
            setConfig['spatialDistributionPDF'], setConfig['spatialDistributionPDFParams'])
        spatialDistributionPDFMode = spatialDistributionPDF.compute_mode()

        processedFractures = []
        logger.debug('placing the longest fracture')
        added = False
        while not added:
            added = self._place_longest_fracture(fractures[0])

        (addedFracture, seed_x, seed_y) = added
        processedFractures.append(addedFracture)
        logger.debug('the longest fracture added')
        numberOfTries = 0
        maxTriesReached = False

        number = 0
        logger.debug('placing rest of fractures')
        for fracture in fractures[1:]:
            theta = fracture['theta']
            if theta < 0:
                theta += 360
            if maxTriesReached:
                break
            isNewFractureAdded = False

            while not isNewFractureAdded:
                if numberOfTries > self.numberOfMaxTries:
                    logger.info("Global max retries reached, stopping all fracture placements.")
                    maxTriesReached = True
                    break

                referenceWithinDomain = False
                while not referenceWithinDomain:
                    if numberOfTries > self.numberOfMaxTries:
                        logger.info("Global max retries reached during domain validation, stopping all fracture placements.")
                        maxTriesReached = True
                        break

                    distance = (spatialDistributionPDF.get_value() - spatialDistributionPDFMode) * random.choice([-1, 1])
                    angle = np.random.uniform(0, 2 * np.pi)
                    new_x_mid = seed_x + distance * np.cos(angle)
                    new_y_mid = seed_y + distance * np.sin(angle)

                    (new_x_start, new_y_start), (new_x_end, new_y_end) = self._fracture_coordinate(
                        fracture, theta, new_x_mid, new_y_mid)
                    if self._is_within_domain(new_x_start, new_y_start) and self._is_within_domain(new_x_end, new_y_end):
                        referenceWithinDomain = True
                    else:
                        numberOfTries += 1

                if maxTriesReached:
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
                        break

                if numberOfTries > self.numberOfMaxTries:
                    logger.info("Global max retries reached, stopping all fracture placements.")
                    maxTriesReached = True
                    break
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
                        'number of tries': numberOfTries,
                    }
                    processedFractures.append(addedFracture)
            if maxTriesReached:
                break

        self.maxtries.append(numberOfTries)
        processedFractures = self.apertureCalculation.get_calculator(processedFractures)
        return processedFractures

    def _generate_fractures(self, setConfig):
        orientationDistributionPDF = OrientationPDFs(
            setConfig['orientationDistributionPDF'], setConfig['orientationDistributionPDFParams'])

        logger.debug('fractureLengthPDFParams["Lmax"]= %s', setConfig['fractureLengthPDFParams']["Lmax"])

        if setConfig['fractureLengthPDF'] == 'Constant':
            logger.debug('Fracture length PDF is constant')
            n = math.ceil(setConfig['I'] * self.ymax * self.xmax / setConfig['fractureLengthPDFParams']['L'])
            fractures = []
            for _ in range(n):
                newFrac = {'fracture length': setConfig['fractureLengthPDFParams']['L']}
                newFrac['theta'] = orientationDistributionPDF.get_value()
                newFrac['set number'] = self.setNumber
                fractures.append(newFrac)
        else:
            fractureLengthPDF = fractureLengthPDFs(setConfig['fractureLengthPDF'], setConfig['fractureLengthPDFParams'])
            fractures = []
            newFrac = {}
            newFrac['fracture length'] = fractureLengthPDF.get_value()
            newFrac['theta'] = orientationDistributionPDF.get_value()
            newFrac['set number'] = self.setNumber
            fractures.append(newFrac)
            while self._compute_intensity(fractures) < setConfig['I']:
                newFrac = {}
                newFrac['fracture length'] = fractureLengthPDF.get_value()
                newFrac['theta'] = orientationDistributionPDF.get_value()
                newFrac['set number'] = self.setNumber
                fractures.append(newFrac)

        return fractures, setConfig

    # Keep old name as alias
    def generateFractures(self, setConfig):
        return self._generate_fractures(setConfig)

    def _compute_intensity(self, fractures):
        total_length = sum([fracture['fracture length'] for fracture in fractures])
        area = self.xmax * self.ymax
        return total_length / area

    def computeIntensity(self, fractures):
        return self._compute_intensity(fractures)

    def _sort_fractures(self, fractures):
        fractures.sort(key=lambda x: x['fracture length'], reverse=True)
        for i, fracture in enumerate(fractures, start=0):
            fracture['number'] = i
        return fractures

    def sortFractures(self, fractures):
        return self._sort_fractures(fractures)

    def _fracture_coordinate(self, fracture, theta, midX, midY):
        half_length = fracture['fracture length'] / 2
        theta_adjusted = 90 - theta
        new_x_start = midX - half_length * np.cos(np.radians(theta_adjusted))
        new_y_start = midY - half_length * np.sin(np.radians(theta_adjusted))
        new_x_end = midX + half_length * np.cos(np.radians(theta_adjusted))
        new_y_end = midY + half_length * np.sin(np.radians(theta_adjusted))
        return (new_x_start, new_y_start), (new_x_end, new_y_end)

    def fractureCoordinate(self, fracture, theta, midX, midY):
        return self._fracture_coordinate(fracture, theta, midX, midY)

    def _is_within_domain(self, x, y):
        return 0 <= x <= self.xmax and 0 <= y <= self.ymax

    def is_within_domain(self, x, y):
        return self._is_within_domain(x, y)

    def _distance_of_well_from_closest_fracture(self, fractures, wellLocation):
        min_distance = float('inf')
        for frac in fractures:
            x_start, y_start = frac['x_start'], frac['y_start']
            x_end, y_end = frac['x_end'], frac['y_end']
            distance = point_to_segment_distance(
                np.array(wellLocation), np.array([x_start, y_start]), np.array([x_end, y_end]))
            min_distance = min(min_distance, distance)
        return min_distance

    def distanceOfWellFromClosetFracture(self, fractures, wellLocation):
        return self._distance_of_well_from_closest_fracture(fractures, wellLocation)

    def _write_input_properties(self, name, apertureCalculationParameters, fractureSets, stressAzimuth, number=0):
        InputPropertiesDir = os.path.join(self.outputDir, name)
        os.makedirs(InputPropertiesDir, exist_ok=True)
        InputPropertiesFile = os.path.join(InputPropertiesDir, f"{number + 1:03}{name}.txt")
        with open(InputPropertiesFile, 'w') as f:
            f.write("domainLengthX: " + str(self.xmax) + '\n')
            f.write("domainLengthY: " + str(self.ymax) + '\n')
            i = 1
            for fracture_set in fractureSets:
                if isinstance(fracture_set, dict):
                    f.write("fractureSet " + str(i) + " : " + json.dumps(fracture_set, indent=4) + '\n')
                else:
                    f.write("fractureSet " + str(i) + " : " + '\n')
                i += 1
            f.write("stressAzimuth: " + str(stressAzimuth) + '\n')
            f.write("fracture aperture " + str(i) + " : " + json.dumps(apertureCalculationParameters, indent=4) + '\n')

    def generateInputPropertiesFile(self, name, apertureCalculationParameters, fractureSets, stressAzimuth, number=0):
        return self._write_input_properties(name, apertureCalculationParameters, fractureSets, stressAzimuth, number)

    def _write_fracture_coordinates(self, name, fractures, number=0):
        coordinatesOutputDir = os.path.join(self.outputDir, name)
        os.makedirs(coordinatesOutputDir, exist_ok=True)
        InputPropertiesFile = os.path.join(coordinatesOutputDir, f"{number + 1:03}{name}.txt")
        with open(InputPropertiesFile, 'w') as fileID:
            for fracture_set in fractures:
                for frac in fracture_set:
                    fileID.write(
                        f"{frac['x_start']:.4f} {frac['y_start']:.4f} {frac['x_end']:.4f} {frac['y_end']:.4f}\n")

    def generateTextFileForFractureCoordinates(self, name, fractures, number=0):
        return self._write_fracture_coordinates(name, fractures, number)

    def _write_fracture_set(self, name, fractures, number=0):
        setOutputDir = os.path.join(self.outputDir, name)
        os.makedirs(setOutputDir, exist_ok=True)
        InputPropertiesFile = os.path.join(setOutputDir, f"{number + 1:03}{name}.txt")
        with open(InputPropertiesFile, 'w') as fileID:
            for fracture_set in fractures:
                for frac in fracture_set:
                    fileID.write(f"{frac['set number']}\n")

    def generateTextFileForFractureSet(self, name, fractures, number=0):
        return self._write_fracture_set(name, fractures, number)

    def _write_fracture_apertures(self, name, fractures, number=0):
        apertureOutputDir = os.path.join(self.outputDir, name)
        os.makedirs(apertureOutputDir, exist_ok=True)
        InputPropertiesFile = os.path.join(apertureOutputDir, f"{number + 1:03}{name}.txt")
        with open(InputPropertiesFile, 'w') as fileID:
            for fracture_set in fractures:
                for frac in fracture_set:
                    fileID.write(f"{frac['fracture aperture']:.7f}\n")

    def generateTextFileForFractureApertures(self, name, fractures, number=0):
        return self._write_fracture_apertures(name, fractures, number)

    def _write_corrected_apertures(self, name, fractures, number=0):
        correctedApertureDir = os.path.join(self.outputDir, str(name))
        os.makedirs(correctedApertureDir, exist_ok=True)
        file_handles = {}
        try:
            for set_index, fracture_set in enumerate(fractures):
                for fracture in fracture_set:
                    for key, value in fracture.items():
                        if key.startswith('correctedAperture'):
                            if key not in file_handles:
                                filePath = os.path.join(correctedApertureDir, f"{key}_{number + 1:03}.txt")
                                file_handles[key] = open(filePath, 'w')
                            file_handles[key].write(f"{value:.7f}\n")
        finally:
            for file in file_handles.values():
                file.close()

    def generateTextFilesForCorrectedApertures(self, name, fractures, number=0):
        return self._write_corrected_apertures(name, fractures, number)

    def _write_output_properties_per_set(self, name, fractureSets, number=0):
        outputFileForOutputPropertiesDir = os.path.join(self.outputDir, name)
        os.makedirs(outputFileForOutputPropertiesDir, exist_ok=True)
        outputPropertiesFile = os.path.join(outputFileForOutputPropertiesDir, f"{number + 1:03}{name}.txt")

        with open(outputPropertiesFile, 'w') as fileID:
            for i, fractures in enumerate(fractureSets):
                intensity = self._compute_intensity(fractures)
                lengths = [frac['fracture length'] for frac in fractures]
                apertures = [frac['fracture aperture'] for frac in fractures]
                minL = min(lengths)
                maxL = max(lengths)
                avgL = sum(lengths) / len(lengths)
                minAperture = min(apertures)
                maxAperture = max(apertures)
                avgAperture = sum(apertures) / len(apertures)
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
                        fileID.write(f"{key} : {value:.3f}\n")
                    else:
                        fileID.write(f"{key} : {value:.3e}\n")
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

    def generateOutputFileForOutputPropertiesPerSet(self, name, fractureSets, number=0):
        return self._write_output_properties_per_set(name, fractureSets, number)

    def _write_overall_properties(self, name, fractureSets, number=0):
        outputFileForOverallPropertiesDir = os.path.join(self.outputDir, name)
        os.makedirs(outputFileForOverallPropertiesDir, exist_ok=True)
        outputPropertiesFile = os.path.join(outputFileForOverallPropertiesDir, f"{number + 1:03}{name}.txt")

        allFractures = [frac for fractures in fractureSets for frac in fractures]
        totalCountFrac = len(allFractures)

        totalIntersection = 0
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
        totalIntensity = self._compute_intensity(allFractures)
        lengths = [frac['fracture length'] for frac in allFractures]
        apertures = [frac['fracture aperture'] for frac in allFractures]
        minLength = min(lengths)
        maxLength = max(lengths)
        avgLength = sum(lengths) / totalCountFrac
        minAperture = min(apertures)
        maxAperture = max(apertures)
        avgAperture = sum(apertures) / totalCountFrac

        wellLocation = (self.xmax / 2, self.ymax / 2)
        minDistanceFromWell = self._distance_of_well_from_closest_fracture(allFractures, wellLocation)

        stressAzimuths = set()
        for frac in allFractures:
            for key in frac:
                if key.startswith('correctedAperture'):
                    stressAzimuths.add(key)

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

    def generateOutputFileForOverallProperties(self, name, fractureSets, number=0):
        return self._write_overall_properties(name, fractureSets, number)

    def _plot_fractures(self, fractureSets, name, number=0):
        figDir = os.path.join(self.outputDir, 'pics')
        os.makedirs(figDir, exist_ok=True)
        figDirFile = os.path.join(figDir, f"{number + 1:03}{name}.png")

        ratio = self.ymax / self.xmax
        plt.figure(figsize=(10, 10 * ratio))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
        setLabels = [f'Set {i + 1}' for i in range(len(fractureSets))]

        for fractures, color, label in zip(fractureSets, colors, setLabels):
            for fracture in fractures:
                x_start, y_start = fracture['x_start'], fracture['y_start']
                x_end, y_end = fracture['x_end'], fracture['y_end']
                plt.plot([x_start, x_end], [y_start, y_end], color=color, label=label)
                label = "_nolegend_"

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

    def plotFractures(self, fractureSets, name, number=0):
        return self._plot_fractures(fractureSets, name, number)

    def _plot_corrected_apertures(self, name, fractureSets, number=0):
        figDir = os.path.join(self.outputDir, 'stressDependency')
        os.makedirs(figDir, exist_ok=True)
        figDirFile = os.path.join(figDir, f"{number + 1:03}{name}.png")

        stress_azimuths = [key for key in fractureSets[0][0].keys() if key.startswith('correctedAperture')]

        num_rows = 2
        num_cols = (len(stress_azimuths) + num_rows - 1) // num_rows
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 18), constrained_layout=True)

        all_apertures = [fracture[key] for fracture_set in fractureSets for fracture in fracture_set for key in stress_azimuths]
        min_aperture, max_aperture = min(all_apertures), max(all_apertures)

        norm = mcolors.Normalize(vmin=min_aperture, vmax=max_aperture)
        cmap = plt.get_cmap('viridis')

        ax = axes.flat[0]
        ax.set_aspect('equal', adjustable='box')
        for set_index, fractures in enumerate(fractureSets):
            for fracture in fractures:
                corrected_aperture = fracture['initial aperture']
                color = cmap(norm(corrected_aperture))
                x_start, y_start = fracture['x_start'], fracture['y_start']
                x_end, y_end = fracture['x_end'], fracture['y_end']
                ax.plot([x_start, x_end], [y_start, y_end], color=color)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Initial aperture')
        ax.grid(True)
        ax.set_xlim(0, self.xmax)
        ax.set_ylim(0, self.ymax)

        for ax_idx, azimuth in enumerate(stress_azimuths):
            ax = axes.flat[ax_idx + 1]
            ax.set_aspect('equal', adjustable='box')
            for set_index, fractures in enumerate(fractureSets):
                for fracture in fractures:
                    corrected_aperture = fracture[azimuth]
                    color = cmap(norm(corrected_aperture))
                    x_start, y_start = fracture['x_start'], fracture['y_start']
                    x_end, y_end = fracture['x_end'], fracture['y_end']
                    ax.plot([x_start, x_end], [y_start, y_end], color=color)
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

    def plotCorrectedApertures(self, name, fractureSets, number=0):
        return self._plot_corrected_apertures(name, fractureSets, number)

    def _plot_orientation_stereographic(self, name, fractureSets, number=0):
        orientationStereographicDir = os.path.join(self.outputDir, name)
        os.makedirs(orientationStereographicDir, exist_ok=True)
        orientationStereographicDirFile = os.path.join(orientationStereographicDir, f"{number + 1:03}{name}.png")
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        fig, ax = plt.subplots(subplot_kw={'polar': True})
        for i, fracture_set in enumerate(fractureSets):
            theta_values = [frac['theta'] for frac in fracture_set]
            theta_radians = np.radians(theta_values)
            ax.hist(theta_radians, bins=36, density=True, alpha=0.75, color=colors[i % len(colors)],
                    label=f'Set {i + 1}')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        plt.legend()
        plt.savefig(orientationStereographicDirFile, format='png', dpi=300)

    def plotOrientationStereographic(self, name, fractureSets, number=0):
        return self._plot_orientation_stereographic(name, fractureSets, number)


def point_to_segment_distance(p, a, b):
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
        return False, None, None

    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant

    if (min(x1, x2) <= x <= max(x1, x2) and
            min(y1, y2) <= y <= max(y1, y2) and
            min(x3, x4) <= x <= max(x3, x4) and
            min(y3, y4) <= y <= max(y3, y4)):
        return True, x, y
    else:
        return False, x, y
