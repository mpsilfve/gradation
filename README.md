# gradation

## 0 



I started with something slightly easier than analyzing the activations in the network, namely analyzing the encoder hidden states. I trained a bidirectional network with 250 hidden states for the nominative -> genitive inflection task. 

When the input is a word form of length N, we get a hidden state tensor of size N x 500. I concatenated that with the negation of itself so that we can find both states which are strongly positive and states which are strongly negative when gradation occurs. This gives a tensor of dimension N x 1000. [I thought this would be better than using absolute value because we are looking for states which behave similarly whenever there is gradation, not something that is strongly negative sometimes and strongly positive other times] 

I found two states which seem to fire strongly when gradation occurs for all three consonants k, p and t. The exact values don't matter of course but they are 993 and 756. 

I checked when the activation of 756 is at its highest in word forms where gradation occurs. Counting from the end of the string, this happens on average at index 1.9 for k, 2.5 for p and 2.7 for t. This might agree fairly well with where we would expect gradation to occur in the word form. I haven't checked if the maximum actually falls at the consonant undergoing gradation yet.

For the other state 993, the picture isn't quite as clear. For k and t, it does attain its maximum near the end of the word but this is not true for p. One thing to note is that we've got fewer examples of p gradation than we have for k or t.  I guess it just happens to be the rarest type overall. 

I think there are confounding factors here which we should investigate. Maybe these just fire whenever there is a k, p or t in the word regardless of gradation. This should be testable. We also need to test how probable it is that this is just random noise. Can you think of other things we need to control for?

There are some things I think would be nice to try. If we just set one of those states to a high value, can we induce the network to perform gradation when it usually wouldn't? I guess that might work but also might not since maybe we'd actually need to perturb some other values as well to ensure that the network won't break. 


## 1


I checked and that 756 state really tends to attain its maximal activation for the consonant which undergoes gradation. Here I've capitalized the position in the input string where the activation is maximal for k, p and t:

 
756 k ['kiukKu', 'arKi', 'nakKi', 'aukKo', 'aikA', 'valtikKa', 'kykY', 'istukKa', 'nukKa', 'merkKi', 'kaksikKo', 'nurkKa', 'elukKa', 'kirsikKa', 'salKo', 'kenKä', 'kreikKa', 'liIke', 'lusikKa', 'sammakKo']

756 p ['sopU', 'prinsiipPi', 'pampPu', 'sinapPi', 'siirapPi', 'tolpPa', 'silpPu']

756 t ['reagoinTi', 'olenTo', 'hajonTa', 'tontTi', 'kentTä', 'viestinTä', 'skyytTi', 'esitelmöinTi', 'liitTo', 'matTi', 'netTi', 'Tarjotin', 'vAikute', 'kantTi', 'tavOite', 'Tiedote', 'minuutTi', 'johTo', 'esIte', 'rinTa', 'attentaatTi', 'kiljunTa', 'isänTä', 'sortTi', 'sekunTi', 'pidätetTy', 'harkinTa', 'koordinoinTi', 'Jäte', 'sentTi', 'Ratas', 'lähTö', 'hänTä', 'laTu', 'silTa', 'sukupuutTo', 'visiitTi', 'kyyTi', 'taiTo', 'lAite', 'veTy', 'synTi', 'mäTi', 'tunTi', 'rAnne', 'kristikunTa', 'monumentTi', 'oluT']
 
 
There are a few cases where it doesn't seem to work like Jäte, lAite and so on. It's noteworthy that many of those are instances where the lemma has weak grade and genitive strong grade which is the more uncommon direction. Usually the lemma has strong grade and genitive form weak grade. 

I also checked the activation for 756 for two other common stem alternations s-d as in avaruus/avaruuden and n-s as in ihminen/ihmisen. For those alternations, the maximal activation is all over the string. So, I think this is at least encouraging evidence of the fact that we might have found a gradation tracking state.

