import numpy as np
import matplotlib.pyplot as plt


def plot(fbpinn, pinn, problem, input, history_fbpinn, history_fbpinn_flops, history_pinn, history_pinn_flops):
    fig = plt.figure(figsize=(15,8))
    grid = plt.GridSpec(3, 4, hspace=0.4, wspace=0.2)

    fbpinn_subdom = fig.add_subplot(grid[0,:2])
    fbpinn_vs_exact = fig.add_subplot(grid[0,2:])
    window_fct = fig.add_subplot(grid[1,0:2])

    pinn_vs_exact = fig.add_subplot(grid[-1,0:2])

    training_error_l2 = fig.add_subplot(grid[-1,-1])
    training_error_flop= fig.add_subplot(grid[-1,-2])


    #plot of FBPiNN with subdomain definition - every subdomain different color

    pred_fbpinn, fbpinn_output, window_output, flops = fbpinn.plotting_data(input)
    for i in range(fbpinn.nwindows):
        fbpinn_subdom.plot(input.detach().numpy(),fbpinn_output[i,].detach().numpy())

    fbpinn_subdom.set_ylabel('u')
    fbpinn_subdom.set_xlabel('x')
    fbpinn_subdom.set_title('FBPiNN: individual network solution')


    #plot of FBPiNN's solution vs exact solution

    fbpinn_vs_exact.plot(input.detach().numpy(), problem.exact_solution(input).detach().numpy(), label="Exact Solution")
    fbpinn_vs_exact.plot(input.detach().numpy(),pred_fbpinn.detach().numpy(), label="Prediction")
    fbpinn_vs_exact.set_ylabel('u')
    fbpinn_vs_exact.set_xlabel('x')
    fbpinn_vs_exact.legend()
    fbpinn_vs_exact.set_title('FBPiNN: global solution vs exact')

    # plot of different PiNN config vs exact solution

    pred, flops = pinn.forward(input)
    pinn_vs_exact.plot(input.detach().numpy(), problem.exact_solution(input).detach().numpy(), label="Exact Solution")
    pinn_vs_exact.plot(input.detach().numpy(),pred.detach().numpy(), label="Prediction")
    pinn_vs_exact.set_ylabel('u')
    pinn_vs_exact.set_xlabel('x')
    pinn_vs_exact.legend()
    pinn_vs_exact.set_title('PiNN: global solution vs exact')


    # Test loss (L1 norm) vs Trainings step

    training_error_l2.plot(np.arange(1, len(history_fbpinn) + 1), history_fbpinn, label="FBPiNN")
    training_error_l2.plot(np.arange(1, len(history_pinn) + 1), history_pinn, label="PiNN ")
    training_error_flop.set_xlabel('Training Step')
    training_error_flop.set_ylabel('Relative L2 error')
    training_error_l2.legend()
    training_error_l2.set_title('Comparing test errors')

    #Test loss (L1 norm) vs FLOPS (floating point operations)

    training_error_flop.plot(history_fbpinn_flops, history_fbpinn, label="FBPiNN")
    training_error_flop.plot(history_pinn_flops, history_pinn, label="PiNN")
    training_error_flop.set_xlabel('FLOPS')
    training_error_flop.set_ylabel('Relative L2 error')
    training_error_flop.legend()
    training_error_flop.set_title('Comparing Test errors vs FLOPs')

    #add-on: cool plot from fig 6 - with subdomain definition and overlap stuff

    if len(fbpinn.manual_part)==0:
        partition = fbpinn.partition_domain()
    else:
        partition = fbpinn.manual_partition()     

    for i in range(fbpinn.nwindows):
        window_fct.hlines( -0.5 if i%2 else -0.4, partition[i][0], partition[i][1],  linewidth=5)
    
    window_fct.hlines(-1, partition[0][0], partition[fbpinn.nwindows-1][1],  linewidth=2, color = 'tab:grey')
    for j in range(fbpinn.nwindows-1):
        window_fct.hlines(-1,partition[j][1], partition[j+1][0],  linewidth=5, color = 'magenta')

    for i in range(fbpinn.nwindows):
        window_fct.plot(input.detach().numpy(),window_output[i,].detach().numpy())


    window_fct.set_yticks([-1,-0.45,0,0.5,1],['overlap','subdomain',0,'window function',1])
    window_fct.set_xlabel('x')
    window_fct.set_title('FBPiNN window function and domains')


    plt.show()