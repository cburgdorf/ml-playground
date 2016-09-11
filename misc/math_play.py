knob_weight = 0.5
input = 0.5
goal_prediction = 0.8

for iteration in range(98):
    prediction = input * knob_weight
    error = (prediction - goal_prediction) ** 2

    direction_and_amount = (goal_prediction - prediction) * input
    knob_weight = knob_weight + direction_and_amount

    print "Loss:" + str(error) + " Prediction:" + str(prediction)









### STEP 1

# prediction = input * knob_weight
# error = (prediction - goal_prediction) ** 2
# print prediction
# print error


### STEP 2

# knob_weight = 0.5
# input = 0.5
# goal_prediction = 0.8
# step_amount = 0.001
#
# for iteration in range(1101):
#     prediction = input * knob_weight
#     error = (prediction - goal_prediction) ** 2
#
#     print "Loss:" + str(error) + " Prediction:" + str(prediction)
#
#     up_prediction = input * (knob_weight + step_amount)
#     up_error = (goal_prediction - up_prediction) ** 2
#
#     down_prediction = input * (knob_weight - step_amount)
#     down_error = (goal_prediction - down_prediction) ** 2
#
#     if(down_error < up_error):
#         knob_weight = knob_weight - step_amount
#
#     if(down_error > up_error):
#         knob_weight = knob_weight + step_amount



### STEP 3

# knob_weight = 0.5
# input = 0.5
# goal_prediction = 0.8
#
# for iteration in range(98):
#     prediction = input * knob_weight
#     error = (prediction - goal_prediction) ** 2
#
#     direction_and_amount = (goal_prediction - prediction) * input
#     knob_weight = knob_weight + direction_and_amount
#
#     print "Loss:" + str(error) + " Prediction:" + str(prediction)
