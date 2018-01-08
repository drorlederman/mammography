import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

    import matplotlib.pyplot as plt

    # make the pie circular by setting the aspect ratio to 1
    plt.figure(figsize=plt.figaspect(1))
    values = [3, 12, 5, 8]
    labels = ['a', 'b', 'c', 'd']

# def make_autopct(labels):
#     def my_autopct(pct):
#         #total = sum(values)
#         #val = int(round(pct * total / 100.0))
#         #print('{}'.format('dror'))
#         #str = '{2f}%  ({d}) {}'.format(p=pct, v=val, s=labels)
#         print(labels)
#         #str = '{l:}({p:2f})'.format(l=labels, p=pct)
#         return '{}({:2f})'.format(labels, pct)

    return my_autopct

# def make_autopct(values, labels):
#     def my_autopct(pct):
#         #total = sum(values)
#         #val = int(round(pct * total / 100.0))
#         #print('{}'.format('dror'))
#         #str = '{2f}%  ({d}) {}'.format(p=pct, v=val, s=labels)
#         print(values)
#         #str = '{l:}({p:2f})'.format(l=labels, p=pct)
#         #return '{p:2f}{v:2f}'.format(p=pct, v=values)
#         return '{p:.2f}% ({v:d})'.format(p=pct, v=values)
#         #return '{}({:2f})'.format(labels, pct)

 #   return my_autopct

def make_autopct(values, labels):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        #return '{p:.2f}% ({v:d})'.format(p=pct, v=values)
    return my_autopct

def CalcPercantage(values):
    total = sum(values)
    per = []
    for v in values:
        per.append(100*v/total)
    return per

# =============================================================================
#  INbreast database
# =============================================================================

# BIRADS histogram:

Bi_rads = pd.read_excel('INbreast.xls',parse_cols='H');
Bi_rads = list(Bi_rads.values.flatten()); # Convert to list

# Convert each 4a\4b\4c values into a value of 4:
Bi_rads = [4 if x == '4a' else x for x in Bi_rads];
Bi_rads = [4 if x == '4b' else x for x in Bi_rads];
Bi_rads = [4 if x == '4c' else x for x in Bi_rads];

Bi_rads = Bi_rads[:len(Bi_rads) - 1]; # Exclude the last row
Bi_rads = np.asarray(Bi_rads); # Convert to array

# ACRs
ACRs = pd.read_excel('INbreast.xls',parse_cols='G');
ACRs = list(ACRs.values.flatten()); # Convert to list
ACRs = ACRs[:len(Bi_rads) - 1]; # Exclude the last row
ACRs = np.asarray(ACRs); # Convert to array

# =============================================================================
#  MIAS database
# =============================================================================


Bi_Rads_MIAS = pd.read_excel('MIAS.xls',parse_cols='B');

Bi_Rads_MIAS = list(Bi_Rads_MIAS.values.flatten()); # Convert to list

# Convert each 4a\4b\4c values into a value of 4:
Bi_Rads_MIAS_no = [1 if x == 'F' else x for x in Bi_Rads_MIAS];
Bi_Rads_MIAS_no = [2 if x == 'G' else x for x in Bi_Rads_MIAS_no];
Bi_Rads_MIAS_no = [3 if x == 'D' else x for x in Bi_Rads_MIAS_no];
Bi_Rads_MIAS_no = np.asarray(Bi_Rads_MIAS_no); # Convert to array

# =============================================================================
# Plot histograms
# Pie

# Corrected 2
# =============================================================================
#
# =============================================================================
# INBreast database
# =============================================================================

# Pie chart

resultsMIAS, edgesMIAS = np.histogram(Bi_Rads_MIAS_no, bins = range(5));
resultsMIASPer = CalcPercantage(resultsMIAS)

resultsINBREAST, edgesINBREAST = np.histogram(ACRs, bins = range(6));
resultsINBREASTPer = CalcPercantage(resultsINBREAST)

resultsINBREAST_Birad, edgesINBREAST_Birad = np.histogram(Bi_rads, bins = range(8));
resultsINBREAST_Birad_Per = CalcPercantage(resultsINBREAST_Birad)

labelsMIAS = []
labelsMIAS.append('Almost entirely \n fatty ({:2.1f}%)'.format(resultsMIASPer[1]))
labelsMIAS.append('Fibroglandular \n density ({:2.1f}%)'.format(resultsMIASPer[2]))
labelsMIAS.append('Heterogeneously \n dense ({:2.1f}%)'.format(resultsMIASPer[3]))

labelsINBREAST = []
labelsINBREAST.append('Almost entirely fatty \n (BI-RAD density 1)({:2.1f}%)'.format(resultsINBREASTPer[1]))
labelsINBREAST.append('Fibroglandular density \n (BI-RAD density 2)({:2.1f}%)'.format(resultsINBREASTPer[2]))
labelsINBREAST.append('Heterogeneously dense \n (BI-RAD density 3)({:2.1f}%)'.format(resultsINBREASTPer[3]))
labelsINBREAST.append('Extremely dense \n (BI-RAD density 4)({:2.1f}%)'.format(resultsINBREASTPer[4]))

labelsINBREAST_Birad = []
labelsINBREAST_Birad.append('BI-RAD 0 ({:2.1f}%)'.format(resultsINBREAST_Birad_Per[1]))
labelsINBREAST_Birad.append('BI-RAD 1 ({:2.1f}%)'.format(resultsINBREAST_Birad_Per[2]))
labelsINBREAST_Birad.append('BI-RAD 2 ({:2.1f}%)'.format(resultsINBREAST_Birad_Per[3]))
labelsINBREAST_Birad.append('BI-RAD 3 ({:2.1f}%)'.format(resultsINBREAST_Birad_Per[4]))
labelsINBREAST_Birad.append('BI-RAD 4 ({:2.1f}%)'.format(resultsINBREAST_Birad_Per[5]))
labelsINBREAST_Birad.append('BI-RAD 5 ({:2.1f}%)'.format(resultsINBREAST_Birad_Per[6]))

colors = ['navy', 'maroon', 'darkmagenta', 'darkslategray','crimson','mediumspringgreen']

colors = ['navy', 'maroon', 'darkmagenta', 'darkslategray','crimson','mediumspringgreen']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','crimson','mediumspringgreen']
#fig1, ax1 = plt.subplots()
#fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)

#fig = plt.figure(1)
fig, ax1 = plt.subplots()
#fig, ax1 = plt.subplots(nrows=1, ncols=2)
#plt.clf()
#fig.clf()
plt.rcParams['font.size'] = 20.0
fig.set_size_inches(16,12)
patches, texts, autotexts = ax1.pie(resultsINBREAST_Birad[1:], colors=colors, labels=labelsINBREAST_Birad, autopct='', startangle=90)
#ax1.text(-0.067, -1.30, '(a)', {'color': 'k', 'fontsize': 16})
ax1.axis('equal')
fig.show()
plt.show()
plt.savefig('INBreast database BIRAD classification');

fig, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
plt.rcParams['font.size'] = 18.0
#fig, ax1 = plt.subplots(figsize=(4,4))
#plt.figure(figsize=(20,10))
#fig, ax1 = plt.subplots()
#fig.set_size_inches(10,8)
#plt.figure(figsize=(20,10)) – StackG Apr 25 '15 at 10:16

#fig, ax = plt.subplots(figsize=(20, 10))
#fig.set_size_inches(16,12)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=None)
patches, texts, autotexts = ax1[0].pie(resultsMIAS[1:], colors=colors, labels=labelsMIAS, autopct='', startangle=90) #, labeldistance=-0.5) #, pctdistance=1.1,
ax1[0].axis('equal')
ax1[0].text(-0.067, -1.30, '(a)', {'color': 'k', 'fontsize': 16})
#fig.show()
#plt.savefig('Mini-MIAS database densities');

#fig, ax1 = plt.subplots()
#fig.set_size_inches(16,12)
ax1[1].axis('equal')
ax1[1].text(-0.067, -1.30, '(b)', {'color': 'k', 'fontsize': 16})
#patches, texts, autotexts = ax2.pie(resultsINBREAST[1:], colors=colors, labels=labelsINBREAST, autopct=make_autopct(labelsINBREAST), startangle=90)
patches, texts, autotexts = ax1[1].pie(resultsINBREAST[1:], colors=colors, labels=labelsINBREAST, autopct='', startangle=0)
#fig.set_size_inches(4, 2, forward=True)
#fig = plt.gcf()
#fig.set_size_inches(2,2) # or (4,4) or (5,5) or whatever
fig.show()
#plt.savefig('INBreast database BIRAD densities');
plt.savefig('Mini-MIAS and INBreast database BIRAD densities');

a=1
#
# fig = plt.figure(1)
# #
# gs = gridspec.GridSpec(2, 2)
# #ax1 = plt.subplot(gs[0, :])
# ax2 = plt.subplot(gs[1, :-1])
# ax3 = plt.subplot(gs[1:, -1])
# #
# # gs1 = gridspec.GridSpec(1, 1)
# # gs1.update(left=0.05, right=0.48, wspace=0.05)
# # ax1 = plt.subplot(gs1[:, :-1])
# # ax2 = plt.subplot(gs1[-1, :-1])
# # ax3 = plt.subplot(gs1[-1, -1])
#
# #gs2 = gridspec.GridSpec(2, 2)
# #gs2.update(left=0.55, right=0.98, hspace=0.05)
# #ax4 = plt.subplot(gs2[-1, :-1])
# #ax5 = plt.subplot(gs2[-1, -1])
#
# #plt.show()
#
# #labels = 'BIRAD 1','BIRAD 2','BIRAD 3','BIRAD 4','BIRAD 5','BIRAD 6'
#
# #patches, texts, autotexts = ax1.pie(resultsINBREAST_Birad[1:], colors=colors, labels=labelsINBREAST_Birad, autopct='', startangle=90)
# #ax1.text(-0.067, -1.30, '(a)', {'color': 'k', 'fontsize': 14})
#
# patches, texts, autotexts = ax2.pie(resultsMIAS[1:], colors=colors, labels=labelsMIAS, autopct='', startangle=90) #, labeldistance=-0.5) #, pctdistance=1.1,
#
# #fig.set_size_inches(0.6,0.6) # or (4,4) or (5,5) or whatever
#
# #fig = plt.gcf()
# #plt.subplot(121)
# #ax1.pie(results[1:], colors=colors, labels=labels, autopct='%1.1f%%', startangle=90)
# # draw circle
# #patches, texts, autotexts = ax1.pie(resultsMIAS[1:], colors=colors, labels=labelsMIAS, autopct=make_autopct(labelsMIAS), startangle=90, pctdistance=1.1, labeldistance=1.2)
#
# # set up subplot grid
#
# # set up subplot grid
#
# #autopct='%1.1f%%'
# #plt.pie(values, labels=labels, autopct=make_autopct(values))
#
# #plt.pie(sizes, labels=labels, autopct='%1.0f%%')
# # for tx in range(len(texts)):
# #     texts[tx].set_fontsize(18)
# # for tx in range(len(autotexts)):
# #     autotexts[tx].set_fontsize(18)
#
# #centre_circle = plt.Circle((0, 0), 0.70, fc='white')
# #ax1.add_artist(centre_circle)
# ax2.axis('equal')
# ax2.text(-0.067, -1.30, '(b)', {'color': 'k', 'fontsize': 14})
#
# #patches, texts, autotexts = ax2.pie(resultsINBREAST[1:], colors=colors, labels=labelsINBREAST, autopct=make_autopct(labelsINBREAST), startangle=90)
# patches, texts, autotexts = ax3.pie(resultsINBREAST[1:], colors=colors, labels=labelsINBREAST, autopct='', startangle=0)
# #centre_circle = plt.Circle((0, 0), 0.70, fc='white')
# #plt.tight_layout()
# plt.show()
#
#
# #ax2.add_artist(centre_circle)
# # Equal aspect ratio ensures that pie is drawn as a circle
# ax3.axis('equal')
# #plt.tight_layout()
# #plt.set_size_inches(0.6,0.6) # or (4,4) or (5,5) or whatever
# ax3.text(-0.067, -1.30, '(c)',{'color': 'k', 'fontsize': 14})
#
# # for tx in range(len(texts)):
# #     texts[tx].set_fontsize(18)
# # for tx in range(len(autotexts)):
# #     autotexts[tx].set_fontsize(18)
#
# plt.show()
#
# # ===========================================================
# # BIRAD Classification scale
# labels = 'BIRAD 1','BIRAD 2','BIRAD 3','BIRAD 4','BIRAD 5','BIRAD 6'
# fig = plt.gcf()
# fig.set_size_inches(0.6,0.6) # or (4,4) or (5,5) or whatever
# patches, texts, autotexts = plt.pie(resultsINBREAST_Birad[1:], colors=colors, labels=labels, autopct='', startangle=90)
# #centre_circle = plt.Circle((0, 0), 0.70, fc='white')
# ax1.axis('equal')
# plt.show()

# ===========================================================


# $$$$$$

#fig1, ax1 = plt.subplots()
#plt.clf()
#plt.subplot(121)

#patches, texts, autotexts = plt.pie(results[1:], labels=labels, colors=colors,
#        autopct='%1.1f%%', shadow=False, startangle=140)


# draw circle
# centre_circle = plt.Circle((0, 0), 0.70, fc='white')
# fig = plt.gcf()
# fig.gca().add_artist(centre_circle)
# # Equal aspect ratio ensures that pie is drawn as a circle
# ax1.axis('equal')
# plt.tight_layout()

#
# # =============================================================================
# # MIAS database
# # =============================================================================
#
# # Pie
# plt.subplot(122)
# labels = 'Fatty','Fatty-glandular','Dense-glandular'
# #F-Fatty
# #G- Fatty - glandular
# #D- Dense - glandular
# #colors = ['navy', 'maroon', 'darkmagenta']
# colors = ['navy', 'maroon','mediumspringgreen']
# # Plot
# results, edges = np.histogram(Bi_Rads_MIAS_no, bins = range(5));
# #patches, texts, autotexts = plt.pie(results[1:], labels=labels, colors=colors,
# #        autopct='%1.1f%%', shadow=False, startangle=140)
# patches, texts, autotexts = ax1.pie(results[1:], colors=colors, labels=labels, autopct='%1.1f%%', startangle=90)
#
# for tx in range(len(texts)):
#     texts[tx].set_fontsize(18)
# for tx in range(len(autotexts)):
#     autotexts[tx].set_fontsize(18)
# # chart of BIRAD score distribution
#
# # draw circle
# centre_circle = plt.Circle((0, 0), 0.70, fc='white')
# #fig = plt.gcf()
# fig.gca().add_artist(centre_circle)
# # Equal aspect ratio ensures that pie is drawn as a circle
# ax1.axis('equal')
# plt.tight_layout()
# plt.show()
# plt.axis('equal')
# plt.savefig('Mini-MIAS database density distribution');
#
# # Data to plot
# #labels = 'Python', 'C++', 'Ruby', 'Java'
# labels = 'BIRAD 1','BIRAD 2','BIRAD 3','BIRAD 4','BIRAD 5','BIRAD 6'
# colors = ['navy', 'maroon', 'darkmagenta', 'darkslategray','crimson','mediumspringgreen']
# #explode = (0.1, 0, 0, 0)  # explode 1st slice
#
# # Plot
# results, edges = np.histogram(ACRs, bins = range(8));
# plt.pie(results[1:], labels=labels, colors=colors,
#         autopct='%1.1f%%', shadow=False, startangle=140)
# # chart of BIRAD score distribution
# plt.axis('equal')
# #plt.title('PIE chart of the BIRAD distribution') # subplot 211 title
# plt.savefig('INBreast database ACR score distribution');
# plt.show()
#
#
# # ============================================================================
# # Data to plot
# #labels = 'Python', 'C++', 'Ruby', 'Java'
# labels = 'BIRAD 1','BIRAD 2','BIRAD 3','BIRAD 4','BIRAD 5','BIRAD 6'
# colors = ['navy', 'maroon', 'darkmagenta', 'darkslategray','crimson','mediumspringgreen']
# #explode = (0.1, 0, 0, 0)  # explode 1st slice
#
# # Plot
# results, edges = np.histogram(Bi_rads, bins = range(8));
# plt.pie(results[1:], labels=labels, colors=colors,
#         autopct='%1.1f%%', shadow=False, startangle=140)
# # chart of BIRAD score distribution
# plt.axis('equal')
# #plt.title('PIE chart of the BIRAD distribution') # subplot 211 title
# plt.savefig('INBreast database BIRAD score distribution');
# plt.show()
#
#
# #
# # # =============================================================================
# #
# # # Regular histogram:
# #
# # results, edges = np.histogram(Bi_rads, bins = range(8));
# # binWidth = 1/1.5;
# # plt.bar(range(7), results, binWidth);
# #
# # plt.title('BIRADS values histogram, INbreast');
# # plt.xlabel('BIRADS values');
# # plt.ylabel('Amount');
# # plt.savefig('BIRADS values histogram, INbreast');
# #
# # # =============================================================================
# #
# # # Percentage histogram:
# #
# # results, edges = np.histogram(Bi_rads, bins = range(8), normed = True);
# # binWidth = 1/1.5;
# # plt.bar(range(7), results, binWidth);
# #
# # # Create the formatter using the function to_percent. This multiplies all the
# # # default labels by 100, making them all percentages
# # formatter = FuncFormatter(to_percent);
# #
# # # Set the formatter
# # plt.gca().yaxis.set_major_formatter(formatter);
# #
# # plt.title('BIRADS values histogram, percentage, INbreast');
# # plt.xlabel('BIRADS values');
# # plt.ylabel('Percent');
# #
# # plt.savefig('BIRADS values histogram, percentage, INbreast');
# # plt.show();
# #
# #
# # #
# # # # ======================================
# # # # plot both densities (mini-MIAS and INBreast together
# # # plt.clf()
# # # plt.subplot(121)
# # #
# # # labels = 'Fatty','Fatty-glandular','Dense-glandular'
# # #
# # # #F-Fatty
# # # #G- Fatty - glandular
# # # #D- Dense - glandular
# # # #colors = ['navy', 'maroon', 'darkmagenta']
# # # colors = ['navy', 'maroon','mediumspringgreen']
# # # # Plot
# # # results, edges = np.histogram(Bi_Rads_MIAS_no, bins = range(5));
# # # #results, edges = np.histogram(Bi_rads, bins = range(8));
# # #
# # # #results, edges = np.histogram(Bi_Rads_MIAS_no);
# # # patches, texts, autotexts = plt.pie(results[1:], labels=labels, colors=colors,
# # #         autopct='%1.1f%%', shadow=False, startangle=140)
# # # for tx in range(len(texts)):
# # #     texts[tx].set_fontsize(18)
# # # for tx in range(len(autotexts)):
# # #     autotexts[tx].set_fontsize(18)
# # # # chart of BIRAD score distribution
# # # plt.axis('equal')
# # # #plt.title('mini-MIAS density distribution') # subplot 211 title
# # #plt.subplot(121)
# # # plt.text(-0.067, -1.14, '(a)', {'color': 'k', 'fontsize': 18})
# # #
# # # plt.subplot(122)
# # #
# # # labels = 'BIRAD 1','BIRAD 2','BIRAD 3','BIRAD 4','BIRAD 5','BIRAD 6'
# # # colors = ['navy', 'maroon', 'darkmagenta', 'darkslategray','crimson','mediumspringgreen']
# # # #explode = (0.1, 0, 0, 0)  # explode 1st slice
# # #
# # # # Plot
# # # results, edges = np.histogram(Bi_rads, bins = range(8));
# # # patches, texts, autotexts = plt.pie(results[1:], labels=labels, colors=colors,
# # #         autopct='%1.1f%%', shadow=False, startangle=140)
# # # for tx in range(len(texts)):
# # #     texts[tx].set_fontsize(18)
# # # for tx in range(len(autotexts)):
# # #     autotexts[tx].set_fontsize(18)
# # #
# # # plt.text(-0.067, -1.14, '(b)',{'color': 'k', 'fontsize': 18})
# # #
# # # # chart of BIRAD score distribution
# # # plt.axis('equal')
# # # #plt.title('INBreast BIRAD score distribution') # subplot 211 title
# # # plt.savefig('BIRAD score distribution');
# # # plt.show()
# #
# #
# # # Corrected figure --------------------------
# # plt.clf()
# # plt.subplot(121)
# #
# # labels = 'Fatty','Fatty-glandular','Dense-glandular'
# #
# # #F-Fatty
# # #G- Fatty - glandular
# # #D- Dense - glandular
# # #colors = ['navy', 'maroon', 'darkmagenta']
# # colors = ['navy', 'maroon','mediumspringgreen']
# # #colors = ['white', 'white','white']
# # # Plot
# # results, edges = np.histogram(Bi_Rads_MIAS_no, bins = range(5));
# # #results, edges = np.histogram(Bi_rads, bins = range(8));
# #
# # #results, edges = np.histogram(Bi_Rads_MIAS_no);
# #
# # fig = plt.figure()
# #
# # patterns = [ "o" , "+" , "x" , "/" , "x", "o", "O", ".", "*" ]
# #
# # #ax1 = fig.add_subplot(111)
# # #for i in range(len(patterns)):
# # #    ax1.bar(i, 3, color='red', edgecolor='black', hatch=patterns[i])
# #
# #
# # plt.show()
# #
# # patches, texts, autotexts = plt.pie(results[1:], labels=labels, colors=colors,
# #         autopct='%1.1f%%', shadow=True, startangle=10, labeldistance=0.8)
# # #patches, texts, autotexts = plt.pie(results[1:], labels=labels, colors=colors,
# # #        autopct='', shadow=True, startangle=10, labeldistance=0.8)
# #
# # fig = plt.figure(1, figsize=(8,8), dpi=60)
# # ax=fig.add_axes([0.1,0.1,0.8,0.8])
# # #labels = ['label0','label1','label2','label3','label4','label5','label6','label7','label8',\
# # #          'label0','label1','label2','label3','label4','label5','label6','label7','label8']
# # #colors = list('w' for _ in range(18))
# # #fracs=list(20 for _ in range(18))
# # #ax.pie(fracs, labels=labels, colors = colors, startangle=10,labeldistance=0.8)
# # #plt.show()
# #
# # #piechart = pie(fracs, explode=explode, autopct='%1.1f%%')
# # #for i in range(len(patches[0])):
# # #    patches[0][i].set_hatch(patterns[(i)%len(patterns)])
# #
# # #for p in range(patches):
# # #for i in range(len(patches)):
# # #    patches[i]._hatch = patterns[(i)%len(patterns)]
# #
# #     #patches, texts = ax.pie(fracs, labels=labels, colors=colors,
# #     #                        startangle=10, labeldistance=0.8)
# # #for t in texts:
# # #    t.set_horizontalalignment('center')
# #
# #
# # plt.show()
# # #for p, pattern in zip(patches, patterns):
# # #    patches = plt.pie(…)[p]
# # #    patches[p].set_hatch(pattern)
# # for tx in range(len(texts)):
# #     texts[tx].set_fontsize(18)
# # for tx in range(len(autotexts)):
# #     autotexts[tx].set_fontsize(18)
# # # chart of BIRAD score distribution
# # plt.axis('equal')
# # #plt.title('mini-MIAS density distribution') # subplot 211 title
# #
# # plt.text(-0.067, -1.14, '(a)', {'color': 'k', 'fontsize': 18})
# #
# # plt.subplot(122)
# #
# # labels = 'Almost entirely \n fatty (ACR 1)','Fibroglandular \n density  (ACR 2)','Heterogeneously \n dense  (ACR 3)','Extremely \n dense  (ACR 4)'
# # colors = ['navy', 'maroon', 'darkmagenta', 'darkslategray']   #,'crimson','mediumspringgreen']
# # #explode = (0.1, 0, 0, 0)  # explode 1st slice
# #
# # # Plot
# # results, edges = np.histogram(ACRs, bins = range(6));
# # patches, texts, autotexts = plt.pie(results[1:], labels=labels,
# #         autopct='%1.1f%%', shadow=False, startangle=140)
# # for tx in range(len(texts)):
# #     texts[tx].set_fontsize(18)
# # for tx in range(len(autotexts)):
# #     autotexts[tx].set_fontsize(18)
# #
# # plt.text(-0.067, -1.14, '(b)',{'color': 'k', 'fontsize': 18})
# #
# # # chart of BIRAD score distribution
# # plt.axis('equal')
# # #plt.title('INBreast BIRAD score distribution') # subplot 211 title
# # plt.savefig('ACR density score distribution');
# # plt.show()
# #
# #
# # # Regular histogram:
# #
# # plt.bar(range(len(Bi_Rads_MIAS)), Bi_Rads_MIAS, align='center');
# #
# # plt.title('BIRADS values histogram, MIAS');
# # plt.xlabel('BIRADS values');
# # plt.ylabel('Amount');
# # plt.savefig('BIRADS values histogram, MIAS');
# #
# # # =============================================================================
# #
# # # Percentage histogram:
# #
# # plt.bar(range(len(Bi_Rads_Perc)), Bi_Rads_Perc, align='center');
# #
# # formatter = FuncFormatter(to_percent);
# # plt.gca().yaxis.set_major_formatter(formatter);
# #
# # plt.title('BIRADS values histogram, percentage, MIAS');
# # plt.xlabel('Bi- Rads values');
# # plt.ylabel('Percent');
# # plt.savefig('BIRADS values histogram, percentage, MIAS');
# #
