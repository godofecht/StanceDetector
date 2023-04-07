import re
import pandas as pd
import numpy as np

#Converting labels to numbers
def label_to_int(label):
  if label   == 'support':
    return 0
  elif label == 'deny':
    return 1
  elif label == 'query':
    return 2
  elif label == 'comment':
    return 3



#Pre-processing Twitter and Reddit Posts to handle URLs and Mentions. 
#Replaces URLs with $URL$ and mentions with $MENTION$
def processText(text):
  text = re.sub(r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?", "$URL$",text.strip())
  text = re.sub(r"(@[A-Za-z0-9]+)", "$MENTION$", text.strip())

  return text

'''Processing all of Twitter and Reddit data frames to 
    1. Get rid of all NaN values
    2. Remove columns not useful for the Model
    3. Process text 
    4. Return a combined frame consisting of both Twitter and Reddit data'''
    
def processStanceData(twitterDf, RedditDf):
  frames = [twitterDf, RedditDf]

  resultDf = pd.concat(frames)                                                      #Concatenating twitter and reddit data
  result1  = resultDf.replace(np.nan, '', regex=True)                               #Getting rid of NaN values

  result1['labelvalue'] = result1.label_x.apply(label_to_int)                       #Converting labels to numbers
  result1['SrcInre']    = result1['inreText'].str.cat(result1['sourceText'],sep=" ")

  data = result1[['text_x', 'id', 'inre_x', 'source_x' ,'label_x','SrcInre', 'labelvalue' ]].copy()


  '''replyText           - the reply post (whose stance towards the target needs to be learnt)
     replyTextId         - the ID of the reply post
     previousText        - the text to which replyText was replied
     sourceText          - the source post of the conversation thread
     label               - the label value assigned to each post
     previoysPlusSrctext - the concatenation of the previousText and the sourceText
     labelValue          - the numberic value assigned to each label'''

  data.columns = ['replyText', 'replyTextId', 'previousText', 'sourceText', 'label', 'previousPlusSrcText', 'labelValue']

  data['pReplyText']           = data.replyText.apply(processText)
  data['pPreviousPlusSrcText'] = data.previousPlusSrcText.apply(processText)
  data['TextSrcInre']          = data['pReplyText'].str.cat(data['pPreviousPlusSrcText'],sep=" ")
  return data
