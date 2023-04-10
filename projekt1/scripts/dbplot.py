# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:23:57 2023

@author: Torbjørn Bak s203555
"""
import numpy as np

def dbplot(classifier, X, y, grid_range, resolution=100):
    ''' Plot decision boundry for given binomial or multinomial classifier '''

    # smoothness of color-coding:
    levels = 100
    # convert from one-out-of-k encoding, if neccessary:
    if np.ndim(y)>1: y = np.argmax(y,1)
    # compute grid range if not given explicitly:
    if grid_range=='auto':
        grid_range = [X.min(0)[0], X.max(0)[0], X.min(0)[1], X.max(0)[1]]
        
    delta_f1 = np.float(grid_range[1]-grid_range[0])/resolution
    delta_f2 = np.float(grid_range[3]-grid_range[2])/resolution
    f1 = np.arange(grid_range[0],grid_range[1],delta_f1)
    f2 = np.arange(grid_range[2],grid_range[3],delta_f2)
    F1, F2 = np.meshgrid(f1, f2)
    C = len(np.unique(y).tolist())
    # adjust color coding:
    if C==2: C_colors = ['b', 'r']; C_legend = ['Class A (y=0)', 'Class B (y=1)']; C_levels = [.5]
    if C==3: C_colors = ['b', 'g', 'r']; C_legend = ['Class A (y=0)', 'Class B (y=1)', 'Class C (y=2)']; C_levels = [.66, 1.34]
    if C==4: C_colors = ['b', 'w', 'y', 'r']; C_legend = ['Class A (y=0)', 'Class B (y=1)', 'Class C (y=2)', 'Class D (y=3)']; C_levels = [.74, 1.5, 2.26]
    if C>4:
        # One way to get class colors for more than 4 classes. Note this may result in illegible figures!
        C_colors=[]
        C_legend=[]
        for c in range(C):
            C_colors.append(plt.cm.jet.__call__(c*255/(C-1))[:3])
            C_legend.append('Class {0}'.format(c))
        C_levels = [.74, 1.5, 2.26]

    coords = np.array( [[f1[i], f2[j]] for i in range(len(f1)) for j in range(len(f2))] )
    values_list = classifier.predict(coords)
    if values_list.shape[0]!=len(f1)*len(f2): values_list = values_list.T
    values = np.reshape(values_list,(len(f1),len(f2))).T
            
    #hold(True)
    for c in range(C):
        cmask = (y==c); plt.plot(X[cmask,0], X[cmask,1], '.', color=C_colors[c], markersize=10)
    plt.title('Model prediction and decision boundary')
    plt.xlabel('Feature 1'); plt.ylabel('Feature 2');
    plt.contour(F1, F2, values, levels=C_levels, colors=['k'], linestyles='dashed')
    plt.contourf(F1, F2, values, levels=np.linspace(values.min(),values.max(),levels), cmap=plt.cm.jet, origin='image')
    plt.colorbar(format='%.1f'); plt.legend(C_legend)
    #hold(False)