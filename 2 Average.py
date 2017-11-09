import numpy as np
import pickle
import csv

def make_features():        #defining a function
    features = {}           #creating a dictionary
    index = 0               #instantiating value

    with open('DataDictionary.csv') as csvfile:         # opening file
        reader = csv.reader(csvfile, dialect='excel', )     #reading through file
        next(reader)                                    #going to next line
        for row in reader:          #for loop to go through all read lines

            if row[0] == "'heatingorsystemtypeid'":     #checking if the value read from file is equal or not to specified location in row
                with open('HeatingOrSystemTypeID.csv') as subfile:  #if above condition is correct open given file
                    subreader = csv.reader(subfile, dialect='excel')    #reading through file
                    subfeatures = {}                                    #creating another dictionary
                    next(subreader)                                     #going to next line and read it    
                    for subrow in subreader:                            #looping through all read data
                        subfeatures[subrow[0]] = index                  #checking if the specified value is equal index (index=0)
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


def make_ides():
    ides = {}
    index = 0

    with open('properties_2016.csv') as csvfile:
        reader = csv.reader(csvfile, dialect='excel')
        next(reader)
        for row in reader:
            ides[row[0]] = index
            index += 1

    return ides


def size_dict(d):
    count = 0
    for feature in d:
        if type(d[feature]) == dict:
            for sub in d[feature]:
                count += 1
        else:
            count += 1
    return count


#Replace all zeros in column number 'col' in 'array' with
#the average of the values in said column
#Output: modified array
def avg_fill(array, col):
    total = 0
    numIdes = 0
    for row in array:
        if row[col] != "":
            total += row[col]
            numIdes += 1 
    average = float(total)/numIdes
    print("average: " + str(average))
    for r in range(size_dict(ides)):
        if array[r, col] == 0.0:
            array[r, col] = average
            print(array[r, col])
    return array

feat = pickle.load(open("feat.p", "rb"))
ides = pickle.load(open("ides.p", "rb"))

"""
feat = make_features() #dictionary of features mapped to column indexes
ides = make_ides() #dictionary of parcel id's mapped to row indexes

pickle.dump(feat, open("feat.p", "wb")) #pickle features dictionary
pickle.dump(ides, open("ides.p", "wb")) #pickle id's dictionary

xvector = np.fromfile('xvector.bin', dtype=float, count=-1, sep='') #load 1D array from bin file
xvector = np.reshape(xvector, (-1, size_dict(feat))) #resize 1D array as 2D array
x = avg_fill(xvector, feat["'taxvaluedollarcnt'"])
x = avg_fill(xvector, feat["'structuretaxvaluedollarcnt'"])
x = avg_fill(xvector, feat["'landtaxvaluedollarcnt'"])
x = avg_fill(xvector, feat["'taxamount'"])
x.tofile('xvector_avg.bin')
"""