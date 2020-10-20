import xlsxwriter
from Tools import *
import os
import subprocess
import time

import sys

#Define Parameters
TerrainName = sys.argv[1]

InitSeedType = sys.argv[2]

ChosenSolver = sys.argv[3]

ShowFigure =  'False'

NumofRound = 14

MinNumLookAhead = 2
MaxNumLookAhead = 10

NumofTrials = 1

ResultFolder = TerrainName + '_' + ChosenSolver + '_' + InitSeedType

if os.path.exists(ResultFolder) == True:
    raise Exception("Same Result Folder exist, clean or backup it, and then delete")
elif os.path.exists(ResultFolder) == False:
    print("Result Saving Path does not exist, create")
    os.mkdir(ResultFolder)

print("Result saved in Path: ", ResultFolder)

workbook = xlsxwriter.Workbook(ResultFolder+'/stats.xlsx')
worksheet = workbook.add_worksheet()

for NumLookAhead in range(MinNumLookAhead,MaxNumLookAhead+1):

    #Write indicator into xlsx and also define reference column for each LookAhead
    row = 0
    col = 0 + (NumLookAhead-1)*6

    worksheet.write(row, col, str(NumLookAhead)+' Step LookAhaed')

    print(NumLookAhead, ' Steps Look Ahead')

    for Trial in range(NumofTrials):

        print('Trial', Trial)
        #Write indicator into xlsx for trials
        row = row + 2 #two lines later

        worksheet.write(row, col, str(Trial)+' Trial')

        stream = os.popen('python3 RecedingHorizon_Planning.py ' + TerrainName + ' ' + InitSeedType + ' ' + ChosenSolver + ' ' + str(NumLookAhead) + ' ' + ShowFigure + ' ' + str(NumofRound) + ' ' + str(Trial) + ' ' + ResultFolder)
        output = stream.readlines()
        
        #write log into text
        file_object = open(ResultFolder+'/'+str(NumLookAhead)+'LookAhead_Trial'+str(Trial), 'a')
        for logIdx in range(len(output)):
            file_object.write(output[logIdx])
        file_object.close()

        #print(output)
        ProgramTime, TotalTime, AccCost, MomentCost, MomentumRateCost, TotalCost, TerminalCost, End_X_pos, End_Y_pos, End_Z_pos = GetStatsFromOutputStrings(output_log = output)

        #Write Program Time and Total Time
        #write heads
        worksheet.write(row, col+1, ChosenSolver+'(P)')
        worksheet.write(row, col+2, ChosenSolver+'(T)')
        
        #write data
        row = row + 1
        for queryIndex in range(len(ProgramTime)):
            worksheet.write(row, col, 'Round ' + str(queryIndex))
            worksheet.write(row, col+1, ProgramTime[queryIndex])
            worksheet.write(row, col+2, TotalTime[queryIndex])
            #Move one more row
            row = row + 1

        #append empty till the max round number
        if len(ProgramTime)-1 < NumofRound - 1:
            for emptyIdx in range(NumofRound - len(ProgramTime)):
                #print(emptyIdx+len(ProgramTime))
                worksheet.write(row, col, 'Round ' + str(emptyIdx+len(ProgramTime)))
                row = row + 1
        
        #Write average time row
        worksheet.write(row, col, 'Averate Time')
        if InitSeedType == 'random':
            ProgramTimeAverage = round(np.mean(ProgramTime),4)
            TotalTimeAverage = round(np.mean(TotalTime),4)
            worksheet.write(row, col+1, ProgramTimeAverage)
            worksheet.write(row, col+2, TotalTimeAverage)
        elif InitSeedType == 'previous':
            if len(ProgramTime)>2:
                ProgramTimeAverage = round(np.mean(ProgramTime[2:]),4)
                TotalTimeAverage = round(np.mean(TotalTime[2:]),4)
                worksheet.write(row, col+1, ProgramTimeAverage)
                worksheet.write(row, col+2, TotalTimeAverage)

        row = row + 1

        #write cost info
        worksheet.write(row, col, 'Acc Cost')
        worksheet.write(row, col+1, AccCost)
        row = row + 1
        worksheet.write(row, col, 'Momentum Cost')
        worksheet.write(row, col+1, MomentCost)
        row = row + 1
        worksheet.write(row, col, 'Momentum Rate Cost')
        worksheet.write(row, col+1, MomentumRateCost)
        row = row + 1
        worksheet.write(row, col, 'Acc AM')
        worksheet.write(row, col+1, AccCost + MomentCost)
        row = row + 1
        worksheet.write(row, col, 'Acc AM Rate')
        worksheet.write(row, col+1, AccCost + MomentumRateCost)
        row = row + 1
        worksheet.write(row, col, 'Total Cost')
        worksheet.write(row, col+1, TotalCost)
        row = row + 1
        worksheet.write(row, col, 'Terminal Cost')
        worksheet.write(row, col+1, TerminalCost)
        row = row + 1
        worksheet.write(row, col, 'Terminal X')
        worksheet.write(row, col+1, End_X_pos)
        row = row + 1
        worksheet.write(row, col, 'Terminal Y')
        worksheet.write(row, col+1, End_Y_pos)
        row = row + 1
        worksheet.write(row, col, 'Terminal Z')
        worksheet.write(row, col+1, End_Z_pos)
        row = row + 1

workbook.close()