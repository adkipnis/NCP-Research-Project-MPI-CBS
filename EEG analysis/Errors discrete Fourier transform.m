%% Preparations
clear all 
cd('C:\Users\Alex\Desktop\8. Semester - NCP\MPI CBS Research Project\Data analysis\EEG analysis')
load('behavior_CSP.mat')
n_subj = 33;

for i=4
acc_temp = B(B(:,1)==i, 3);

fs=1/713; %average sampling frequency of response
      acc_X = fft(acc_temp) / size(acc_temp, 1); 
      hz = linspace( 0, fs/2, floor(length(acc_temp)/2)+1 ); % frequencies for which the FFT will give us amplitudes; +1 for DC component
      %[~, minidx] = min(abs(hz-optload.bfreq(1))); %find index of frequency closest to minimum threshold freq
      %[~, maxidx] = min(abs(hz-optload.bfreq(2))); %find index of frequency closest to maximum threshold freq
      %hz = hz(minidx:maxidx); %this gives us all frequencies until the frequency in Hz specified in optload.nfreqs
      %signalX = signalX(minidx:maxidx,:,:); %only positive frequencies from bfreq (previously only all positive freqs)
      %acc_X = acc_X(1:length(hz),:,:); %only positive frequencies from bfreq (previously only all positive freqs)
          amplitudes(1,:,:) = abs(acc_X(1,:,:)); %do not double 0 Hz component
          amplitudes(2:length(acc_temp),:,:) = 2*abs(acc_X(2:end,:,:)); %double all positive freqs as we discard all negative freqs
      
      
 %bin     
 freq_bins = [1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10]
 freq_bins = freq_bins.^(-1);
 
 
      for i=1:length(freq_bins)
       idx{i,:} = find(hz > freq_bins(i,1) & hz <= freq_bins(i,2)); %find indices of values belonging to each bin
       fe(i,:,:) = sum(acc_X(idx{i,:},:,:), 1); %sum values in each bin and save the sum as a new feature
      end
      
 % plot
      plot(hz,amplitudes,'p-')
      title('Frequency domain')
      ylabel('Amplitude'), xlabel('Frequency (Hz)')
end