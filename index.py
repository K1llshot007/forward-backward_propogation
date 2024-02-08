import numpy as np
np.set_printoptions(suppress=True, precision=6)

def forw_relu(matrix):

    resultingMatrix = []

    # using np max between 0 and 1 we remove all the negative values and then give out the new array, memory could be saved here but dont want to touch it
    resultingMatrix.append(np.maximum(0, matrix))
    
    return resultingMatrix

def forw_maxpool(matrix):

    
    row = 0
    col = 0
    ansMatrix = np.zeros(((matrix.shape[0]//2), (matrix.shape[1]//2)))
    # this loop uses stride length 2 and takes the filter across, at every step the max of 2x2 matrix is calculated and then input in another ans matrix which is then output
    for x in range(matrix.shape[0]//2):
        for y in range(matrix.shape[1]//2):
            ansMatrix[x][y] = max(matrix[row][col], matrix[row][col+1], matrix[row+1][col], matrix[row+1][col+1])
            col += 2
        row += 2
        col=0

    
    return ansMatrix

def forw_meanpool(matrix):

    row = 0
    col = 0
    ansMatrix = np.zeros(((matrix.shape[0]//2), (matrix.shape[1]//2)), dtype=np.float32)
    # does the same things as maxpool but instead of taking max takes the mean
    for x in range(matrix.shape[0]//2):
        for y in range(matrix.shape[1]//2):
            ansMatrix[x][y] = (matrix[row][col] + matrix[row][col+1] + matrix[row+1][col] + matrix[row+1][col+1])/4
            col += 2
        row += 2
        col=0

    
    return ansMatrix

def forw_fc(a, weight, b):
    input = a
    weights = weight
    bias = b

    ans = 0
    # just goes through the matrix and does the weight * input, making it scalar and then add a bias to it
    for x in range(input.shape[0]):
        for y in range(input.shape[1]):
            ans += input[x][y] * weights[x][y]
    
    ans += bias

    return ans

def forw_softmax(matrix):
    denominator =0
    # first take the average and make the  denominator
    ans = np.zeros((matrix.shape[0],1),dtype=np.float32)
    for x in range(matrix.shape[0]):
        denominator +=  np.exp(matrix[x])

    # now divide each element by the denominator
    for y in range(matrix.shape[0]):
        ans[y] = np.exp(matrix[y])/(denominator) 
    return ans

def back_relu(input, output, dzdy):

    ans = np.zeros((input.shape[0],input.shape[1]), dtype=np.float32)
    # if  the number in imput is greater than zero make it one otherwise make it zero and then multiply each element by dzdy
    for x in range(input.shape[0]):
        for y in range(input.shape[1]):
            if output[x][y] > 0 :
                output[x][y] = 1.0000

            ans[x][y] = output[x][y] * dzdy[x][y]

    return ans

def back_maxpool(input, output, dzdy):
    ans = np.zeros((input.shape[0],input.shape[1]), dtype=np.float32)
    row =0
    col =0
    #  find the max value index, and then make the value of that index 1 and make the rest 0, then multiply that value with the corresponding dzdy value as only that value will contribute
    for x in range(input.shape[0]//2):
            for y in range(input.shape[1]//2):
                if output[x][y] == input[row][col]:
                    ans[row][col] = 1.0000*dzdy[x][y]
                    ans[row][col+1] = 0
                    ans[row+1][col] = 0
                    ans[row+1][col+1] = 0
                elif output[x][y] == input[row][col+1]:
                    ans[row][col] = 0
                    ans[row][col+1] = 1.0000*dzdy[x][y]
                    ans[row+1][col] = 0
                    ans[row+1][col+1] = 0
                elif output[x][y] == input[row+1][col]:
                    ans[row][col] = 0
                    ans[row][col+1] = 0
                    ans[row+1][col] = 1.0000*dzdy[x][y]
                    ans[row+1][col+1] = 0
                elif output[x][y] == input[row+1][col+1]:
                    ans[row][col] = 0
                    ans[row][col+1] = 0
                    ans[row+1][col] = 0
                    ans[row+1][col+1] = 1.0000*dzdy[x][y]

                col += 2
            row += 2
            col=0

    

    return ans

def back_meanpool(input, output, dzdy):
    ans = np.zeros((input.shape[0],input.shape[1]), dtype=np.float32)
    # multiply the filter area with 0.25 as we are using 2x2 filter
    for x in range(input.shape[0]):
            for y in range(input.shape[1]):
                ans[x][y] = 0.25*dzdy[x//2][y//2]

    return ans

def back_fc(input, weight, bias, output, dzdy):
    dzdw = np.zeros((weight.shape[0],weight.shape[1]), dtype=np.float32)
    dzdx = np.zeros((input.shape[0],input.shape[1]), dtype=np.float32)
    dzdb = np.zeros((1,1), dtype=np.float32)
    dzdb[0][0] = dzdy[0][0]
    # according to the formulas described in the derivations take appropriate products and then return answer
    for x in range(input.shape[0]):
        for y in range(input.shape[1]):
            dzdw[x][y] = dzdy[0][0]*input[x][y]
            dzdx[x][y] = dzdy[0][0]*weight[x][y]
            
    ans = []
    ans.append(dzdx)
    ans.append(dzdw)
    ans.append(dzdb)
    return ans

def back_softmax(input, output, dzdy):
    ans = np.zeros((input.shape[0],input.shape[1]), dtype=np.float32)

    row = 0
    col =0
    # this is if dzdx1 = dzdy1*y1(1-y1) + dzdy2*(-y1*y2) + dzdy3*(-y1*y3) + ..., according to this, the values were added and calculated
    for row in  range(output.shape[0]):
        for y in range(output.shape[0]):

            if row == y:
                ans[row][0] += dzdy[y][0] *(1-output[y][0])*output[y][0]
            if row !=y :
                ans[row][col] += dzdy[y][0]*(output[y][col])*(-1)*output[row][col]
        
        

    

    return ans

            








    

    
def formMatrix(data):
    rows = int(data[0])

    cols = int(data[1])

    matrix = np.zeros((rows,cols),dtype=np.float32)

    x = 2

    #loop below creates the matrix
    for  i in range (cols):
        for j in range(rows):
            matrix[j][i] = float(data[x])
            x+=1


    return matrix

def understandingInput(filename):
    with open(filename, "r") as input:
        file = input.readlines()
        
    lines = 0
    

    while lines<len(file):

        
        if file[lines][0] == 'f':

            
            if file[lines].strip() == "forw_relu":
                #going to number of inputs
                lines+=1 
                numInputs = int(file[lines].strip())

                #going to the rows, columns, and data for input, since input can be > 1, thus will use a for loop and store matrices in a list
                inputMatrices = []
                lines+=1
                for  i in range(numInputs):
                    inputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1 

                for i in inputMatrices:
                    result = forw_relu(i)
                    # print(result)

                # going to number of outputs 
                numOutputs = int(file[lines].strip())

                #going to the rows, columns, and data for output
                lines+=1
                outputMatrices = []

                for  i in range(numOutputs):
                    outputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1

                # print(outputMatrices)



                
            elif file[lines].strip() == "forw_maxpool":
                #going to number of inputs
                lines+=1 
                numInputs = int(file[lines].strip())

                #going to the rows, columns, and data for input, since input can be > 1, thus will use a for loop and store matrices in a list
                inputMatrices = []
                lines+=1
                for  i in range(numInputs):
                    inputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1 

                for i in inputMatrices:
                    result = forw_maxpool(i)
                
                    # print(result)




                # going to number of outputs 
                numOutputs = int(file[lines].strip())

                #going to the rows, columns, and data for output
                lines+=1
                outputMatrices = []

                for  i in range(numOutputs):
                    outputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1
                
                # print(outputMatrices)




            elif file[lines].strip() == "forw_meanpool":
                #going to number of inputs
                lines+=1 
                numInputs = int(file[lines].strip())

                #going to the rows, columns, and data for input, since input can be > 1, thus will use a for loop and store matrices in a list
                inputMatrices = []
                lines+=1
                for  i in range(numInputs):
                    inputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1 

                for i in inputMatrices:
                    result = forw_meanpool(i)
                    # print(result)


                # going to number of outputs 
                numOutputs = int(file[lines].strip())

                #going to the rows, columns, and data for output
                lines+=1
                outputMatrices = []

                for  i in range(numOutputs):
                    outputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1
                
                # print(outputMatrices)



            
            elif file[lines].strip() == "forw_softmax":
                #going to number of inputs
                lines+=1 
                numInputs = int(file[lines].strip())

                #going to the rows, columns, and data for input, since input can be > 1, thus will use a for loop and store matrices in a list
                inputMatrices = []
                lines+=1
                for  i in range(numInputs):
                    inputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1 

                for i in inputMatrices:
                    result = forw_softmax(i)
                    # print(result)


                # going to number of outputs 
                numOutputs = int(file[lines].strip())

                #going to the rows, columns, and data for output
                lines+=1
                outputMatrices = []

                for  i in range(numOutputs):
                    outputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1
                
                # print(outputMatrices)
            
            


            elif file[lines].strip() == "forw_fc":
                #going to number of inputs
                lines+=1 
                numInputs = int(file[lines].strip())

                #going to the rows, columns, and data for input, since input can be > 1, thus will use a for loop and store matrices in a list
                inputMatrices = []
                lines+=1
                for  i in range(numInputs):
                    inputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1 

                result = forw_fc(inputMatrices[0], inputMatrices[1], inputMatrices[2])
                # print(result)


                # going to number of outputs 
                numOutputs = int(file[lines].strip())

                #going to the rows, columns, and data for output
                lines+=1
                outputMatrices = []

                for  i in range(numOutputs):
                    outputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1
                
                # print(outputMatrices)


        else:
            
            if file[lines].strip() == "back_relu":
                #going to number of inputs
                lines+=1 
                numInputs = int(file[lines].strip())

                #going to the rows, columns, and data for input, since input can be > 1, thus will use a for loop and store matrices in a list
                inputMatrices = []
                lines+=1
                for  i in range(numInputs):
                    inputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1 

                result = back_relu(inputMatrices[0], inputMatrices[1], inputMatrices[2])
                print(result)
                # going to number of outputs 
                numOutputs = int(file[lines].strip())

                #going to the rows, columns, and data for output
                lines+=1
                outputMatrices = []

                for  i in range(numOutputs):
                    outputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1
                print(outputMatrices)
                




            elif file[lines].strip() == "back_maxpool":
                #going to number of inputs
                lines+=1 
                numInputs = int(file[lines].strip())

                #going to the rows, columns, and data for input, since input can be > 1, thus will use a for loop and store matrices in a list
                inputMatrices = []
                lines+=1
                for  i in range(numInputs):
                    inputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1 

                
                result = back_maxpool(inputMatrices[0], inputMatrices[1], inputMatrices[2])
                print(result)


                # going to number of outputs 
                numOutputs = int(file[lines].strip())

                #going to the rows, columns, and data for output
                lines+=1
                outputMatrices = []

                for  i in range(numOutputs):
                    outputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1
                print(outputMatrices)




            elif file[lines].strip() == "back_meanpool":
                #going to number of inputs
                lines+=1 
                numInputs = int(file[lines].strip())

                #going to the rows, columns, and data for input, since input can be > 1, thus will use a for loop and store matrices in a list
                inputMatrices = []
                lines+=1
                for  i in range(numInputs):
                    inputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1 

                result = back_meanpool(inputMatrices[0],inputMatrices[1],inputMatrices[2])

                print(result)

                # going to number of outputs 
                numOutputs = int(file[lines].strip())

                #going to the rows, columns, and data for output
                lines+=1
                outputMatrices = []

                for  i in range(numOutputs):
                    outputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1
                
                print(outputMatrices)



            
            elif file[lines].strip() == "back_softmax":
                #going to number of inputs
                lines+=1 
                numInputs = int(file[lines].strip())

                #going to the rows, columns, and data for input, since input can be > 1, thus will use a for loop and store matrices in a list
                inputMatrices = []
                lines+=1
                for  i in range(numInputs):
                    inputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1 

                result = back_softmax(inputMatrices[0],inputMatrices[1],inputMatrices[2] )
                # print(result)


                # going to number of outputs 
                numOutputs = int(file[lines].strip())

                #going to the rows, columns, and data for output
                lines+=1
                outputMatrices = []

                for  i in range(numOutputs):
                    outputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1
                
                # print(outputMatrices)
            
            
            
            
            
            elif file[lines].strip() == "back_fc":
                #going to number of inputs
                lines+=1 
                numInputs = int(file[lines].strip())

                #going to the rows, columns, and data for input, since input can be > 1, thus will use a for loop and store matrices in a list
                inputMatrices = []
                lines+=1
                for  i in range(numInputs):
                    inputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1 

                result = back_fc(inputMatrices[0], inputMatrices[1], inputMatrices[2], inputMatrices[3], inputMatrices[4])

                # print(result)

                # going to number of outputs 
                numOutputs = int(file[lines].strip())

                #going to the rows, columns, and data for output
                lines+=1
                outputMatrices = []

                for  i in range(numOutputs):
                    outputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1
                
                # print(outputMatrices[2])



    
            

understandingInput('hw3testfile.txt') 


