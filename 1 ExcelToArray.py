# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:30:54 2017

@author: kelby
"""

import numpy as np
import csv

def make_features():
    features = {}
    index = 0

    with open('DataDictionary.csv') as csvfile:
        reader = csv.reader(csvfile, dialect='excel')
        next(reader)
        for row in reader:

            if row[0] == "'heatingorsystemtypeid'":
                with open('HeatingOrSystemTypeID.csv') as subfile:
                    subreader = csv.reader(subfile, dialect='excel')
                    subfeatures = {}
                    next(subreader)
                    for subrow in subreader:
                        subfeatures[subrow[0]] = index
                        index += 1
                features[row[0]] = subfeatures

            elif row[0] == "'propertylandusetypeid'":
                with open('PropertyLandUseTypeID.csv') as subfile:
                    subreader = csv.reader(subfile, dialect='excel')
                    subfeatures = {}
                    next(subreader)
                    for subrow in subreader:
                        subfeatures[subrow[0]] = index
                        index += 1
                features[row[0]] = subfeatures

            elif row[0] == "'storytypeid'":
                with open('StoryTypeID.csv') as subfile:
                    subreader = csv.reader(subfile, dialect='excel')
                    subfeatures = {}
                    next(subreader)
                    for subrow in subreader:
                        subfeatures[subrow[0]] = index
                        index += 1
                features[row[0]] = subfeatures

            elif row[0] == "'airconditioningtypeid'":
                with open('AirConditioningTypeID.csv') as subfile:
                    subreader = csv.reader(subfile, dialect='excel')
                    subfeatures = {}
                    next(subreader)
                    for subrow in subreader:
                        subfeatures[subrow[0]] = index
                        index += 1
                features[row[0]] = subfeatures

            elif row[0] == "'architecturalstyletypeid'":
                with open('ArchitecturalStyleTypeID.csv') as subfile:
                    subreader = csv.reader(subfile, dialect='excel')
                    subfeatures = {}
                    next(subreader)
                    for subrow in subreader:
                        subfeatures[subrow[0]] = index
                        index += 1
                features[row[0]] = subfeatures

            elif row[0] == "'typeconstructiontypeid'":
                with open('TypeConstructionTypeID.csv') as subfile:
                    subreader = csv.reader(subfile, dialect='excel')
                    subfeatures = {}
                    next(subreader)
                    for subrow in subreader:
                        subfeatures[subrow[0]] = index
                        index += 1
                features[row[0]] = subfeatures

            elif row[0] == "'buildingclasstypeid'":
                with open('BuildingClassTypeID.csv') as subfile:
                    subreader = csv.reader(subfile, dialect='excel')
                    subfeatures = {}
                    next(subreader)
                    for subrow in subreader:
                        subfeatures[subrow[0]] = index
                        index += 1
                features[row[0]] = subfeatures

            else:
                features[row[0]] = index
                index += 1

    return features


def size_features():
    count = 0
    for feature in features:
        if type(features[feature]) == dict:
            for sub in features[feature]:
                count += 1
        else:
            count += 1
    return count


def print_features(features):
    for feature in features:
        if type(features[feature]) == dict:
            for sub in features[feature]:
                print(sub + " : " + str(features[feature][sub]))
        else:
            print(feature + " : " + str(features[feature]))

"""
Files:
    DataDictionary
    HeatingOrSystemTypeID
    PropertyLandUseTypeID
    StoryTypeID
    AirConditioningTypeID
    ArchitecturalStyleTypeID
    TypeConstructionTypeID
    BuildingClassTypeID
"""

#--------------make_xvector---------------
#Goes through excel sheet column by column and organizes data into matrix
#Input: dictionary with <feature, matrixColumnIndex> mapping
#Output: ID-by-feature matrix of housing data

def make_xvector(d):
    x = np.zeros((2985217, size_features()))

    for col in range(1, 58):
        with open('properties_2016.csv') as csvfile:
            reader = csv.reader(csvfile, dialect='excel')
            feat = "'" + next(reader)[col] + "'"
            rowNum = 0
            
            if feat != "'propertycountylandusecode'" and feat != "'propertyzoningdesc'":
            
                if type(d[feat]) == dict: #if this column's feature has subfeatures
                    for row in reader:
                        data = row[col]
                        if data != "":
                            colNum = d[feat][data]
                            x[rowNum, colNum] = 1
                            print(feat)
                        rowNum += 1
    
                else:
                    for row in reader:
                        data = row[col]
                        if data != "":
                            if data == "true" or data == "Y":
                                data = 1
                            colNum = d[feat]
                            x[rowNum, colNum] = data
                            print(feat)
                        rowNum += 1
                        
    return x


features = make_features() #generate dictionary of feature types
x = make_xvector(features)
x.tofile('xvector.bin')
input("press enter to exit")
