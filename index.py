import numpy as np

def forw_relu(matrix):
    # take the array from the input and multiply it with the weights then add the bias and keep it all in a variable result, so result = weight * activation + bias, then use the squasher on the result to send it to the next layer, so sqausher(result), lets say the squasher is relu, then relu = max(0, result)
    # print("Hellow World ")

    resultingMatrix = []

    
    resultingMatrix.append(np.maximum(0, matrix))
    
    return resultingMatrix

def forw_maxpool(matrix):

    
    row = 0
    col = 0
    ansMatrix = np.zeros(((matrix.shape[0]//2), (matrix.shape[1]//2)))
    for x in range(matrix.shape[0]//2):
        for y in range(matrix.shape[1]//2):
            ansMatrix[x][y] = max(matrix[row][col], matrix[row][col+1], matrix[row+1][col], matrix[row+1][col+1])
            col += 2
        row += 2
        col=0

    
    return ansMatrix

def forw_meanpool(matrix):#working and giving correct answers but if is giving incorrect result
    # resultingMatrices = []

    # for i in matrices:
    row = 0
    col = 0
    ansMatrix = np.zeros(((matrix.shape[0]//2), (matrix.shape[1]//2)), dtype=np.float32)
    for x in range(matrix.shape[0]//2):
        for y in range(matrix.shape[1]//2):
            # ansMatrix[x][y] = np.mean((matrix[row][col] , matrix[row][col+1] , matrix[row+1][col] , matrix[row+1][col+1]), dtype=np.float32)
            ansMatrix[x][y] = (matrix[row][col] + matrix[row][col+1] + matrix[row+1][col] + matrix[row+1][col+1])/4
            col += 2
        row += 2
        col=0
    # resultingMatrices.append(ansMatrix)    

    
    return ansMatrix

def forw_fc(a, weight, b):#working  fine but same thing float 32 gives answers which are very close, like 0.001 close
    # resultingMatrices = []
    
    input = a
    weights = weight
    bias = b

    ans = 0
    
    # dotProduct = np.dot(input,weights)
    # print(dotProduct)
    for x in range(input.shape[0]):
        for y in range(input.shape[1]):
            ans += input[x][y] * weights[x][y]
    
    ans += bias

    # resultingMatrices.append(ans)?
    # print('this is from the function', ans)
    return ans

def forw_softmax(matrix):#same thing to all the above calculations, they calculate to some other degree of accuracy, which is not in the output
    
    denominator =0
    
    ans = np.zeros((matrix.shape[0],1),dtype=np.float32)
    for x in range(matrix.shape[0]):
        denominator +=  np.exp(matrix[x])


    for y in range(matrix.shape[0]):
        ans[y] = np.exp(matrix[y])/(denominator) 
    print('from the function',ans)
    return ans

def back_relu(input, output, dzdy):

    ans = np.zeros((input.shape[0],input.shape[1]), dtype=np.float32)

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

    # for x in range(input.shape[0]):
    #         for y in range(input.shape[1]):
    #             ans[x][y] = input[x][y]*dzdy[x//2][y//2]

    return ans

def back_meanpool(input, output, dzdy):
    ans = np.zeros((input.shape[0],input.shape[1]), dtype=np.float32)

    for x in range(input.shape[0]):
            for y in range(input.shape[1]):
                ans[x][y] = 0.25*dzdy[x//2][y//2]

    return ans

def back_fc(input, weight, bias, output, dzdy):
    dzdw = np.zeros((weight.shape[0],weight.shape[1]), dtype=np.float32)
    dzdx = np.zeros((input.shape[0],input.shape[1]), dtype=np.float32)
    dzdb = np.zeros((1,1), dtype=np.float32)
    dzdb[0][0] = dzdy[0][0]

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


            








#all understanding input function are below, these function help parse the input, check which sqausher or which type of function if being used, forms the input array and sends it to the correct function
    

    
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
    

    # print(file)
    while lines<len(file):
        # print(file[lines])
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

                # print(inputMatrices)
                for i in inputMatrices:
                    result = forw_relu(i)
                    # print("result", result)

                # going to number of outputs 
                numOutputs = int(file[lines].strip())

                #going to the rows, columns, and data for output
                lines+=1
                outputMatrices = []

                for  i in range(numOutputs):
                    outputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1

                # print(lines)
                # print(outputMatrices)
                # for i in range(len(outputMatrices)):
                #     if(outputMatrices[i] == result[i]).all():
                #         print('forward relu working')



                
            elif file[lines].strip() == "forw_maxpool":
                print('hello')
                #going to number of inputs
                lines+=1 
                numInputs = int(file[lines].strip())

                #going to the rows, columns, and data for input, since input can be > 1, thus will use a for loop and store matrices in a list
                inputMatrices = []
                lines+=1
                for  i in range(numInputs):
                    inputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1 

                # print(inputMatrices)
                for i in inputMatrices:
                    result = forw_maxpool(i)
                
                print(result)



                # print(result)

                # going to number of outputs 
                numOutputs = int(file[lines].strip())

                #going to the rows, columns, and data for output
                lines+=1
                outputMatrices = []

                for  i in range(numOutputs):
                    outputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1
                
                print(outputMatrices)
                # for i in range(len(outputMatrices)):
                #     if(outputMatrices[i] == result[i]).all():
                #         print('forward maxpool working')




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

                # print(inputMatrices)
                for i in inputMatrices:
                    result = forw_meanpool(i)
                    # print(result)
                # result = forw_meanpool(inputMatrices)

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
                # for i in range(len(outputMatrices)):
                #     if(outputMatrices[i] == result[i]).all():
                #         print('forward meanpool working')
                #     else:
                #         print("Mean pooling failed")



            
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

                print(inputMatrices)
                for i in inputMatrices:
                    result = forw_softmax(i)
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
            
            
            
            
            
            elif file[lines].strip() == "forw_fc":#working as stated in pdf for the api call
                #going to number of inputs
                lines+=1 
                numInputs = int(file[lines].strip())

                #going to the rows, columns, and data for input, since input can be > 1, thus will use a for loop and store matrices in a list
                inputMatrices = []
                lines+=1
                for  i in range(numInputs):
                    inputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1 

                # print(inputMatrices[2])
                # for i in inputMatrices:
                # print('input 1',inputMatrices[0], 'input2', inputMatrices[1], 'input3',inputMatrices[1])
                result = forw_fc(inputMatrices[0], inputMatrices[1], inputMatrices[2])
                # print('result' ,result)
                # result = forw_fc(inputMatrices)

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
                # for i in range(len(outputMatrices)):
                #     if(outputMatrices[i] == result[i]).all():
                #         print('forward fc working')
                #     else:
                #         print("fc failed")


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

                # print(inputMatrices)
                    # for i in inputMatrices:
                result = back_relu(inputMatrices[0], inputMatrices[1], inputMatrices[2])
                # print(result)
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
                # for i in range(len(outputMatrices)):
                #     if(outputMatrices[i] == result[i]).all():
                #         print('backward relu working')




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

                # print(inputMatrices)
                
                result = back_maxpool(inputMatrices[0], inputMatrices[1], inputMatrices[2])

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
                # for i in range(len(outputMatrices)):
                #     if(outputMatrices[i] == result[i]).all():
                #         print('forward maxpool working')




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

                # print(inputMatrices)
                result = back_meanpool(inputMatrices[0],inputMatrices[1],inputMatrices[2])

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
                # for i in range(len(outputMatrices)):
                #     if(outputMatrices[i] == result[i]).all():
                #         print('forward meanpool working')
                #     else:
                #         print("Mean pooling failed")



            
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

                # print(inputMatrices)
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
                
                print(outputMatrices)
            
            
            
            
            
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

                # print(inputMatrices)
                result = back_fc(inputMatrices[0], inputMatrices[1], inputMatrices[2], inputMatrices[3], inputMatrices[4])

                # print(result[2])

                # going to number of outputs 
                numOutputs = int(file[lines].strip())

                #going to the rows, columns, and data for output
                lines+=1
                outputMatrices = []

                for  i in range(numOutputs):
                    outputMatrices.append(formMatrix(file[lines].strip().split(' ')))  
                    lines+=1
                
                # print(outputMatrices[2])
                # for i in range(len(outputMatrices)):
                #     if(outputMatrices[i] == result[i]).all():
                #         print('forward fc working')
                #     else:
                #         print("fc failed")



    
            lines+=1

understandingInput('hw3testfile.txt') 


