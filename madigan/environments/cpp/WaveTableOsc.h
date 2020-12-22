#ifndef WAVETABLEOSC_H_
#define WAVETABLEOSC_H_

#include <vector>
#include <math.h>

#define constantRatioLimit (99999)

namespace madigan{


  // WaveTable and WaveTableOsc are templated to provide
  // either float or double versions

  template<class T>
  struct WaveTable{
    double topFreq;
    int len;
    T *waveTable;
  };


  template<class T>
  class WaveTableOsc{
  public:
    WaveTableOsc();
    WaveTableOsc(int maxWaveTables);
    ~WaveTableOsc();
    void setFreq(double incr);
    void setPhaseOffset(double offset){ phaseOffset=offset;}
    void updatePhase(){ phasor += phaseIncr; if (phasor >= 1.) phasor -= 1.;}
    T getOutput();
    T process();
    int addWaveTable(int len, T* wave, double top_freq);
    const std::vector<WaveTable<T>>& getWaveTables(){ return waveTables;}

  private:
    double phasor;
    double phaseIncr;
    double phaseOffset=0.5; // for PWM

    int numWaveTables;
    int maxWaveTables=32;
    std::vector<WaveTable<T>> waveTables;
    // WaveTable wavetables[max_wavetables];

    int currWaveTable=0;
  };



  template<class T>
  inline WaveTableOsc<T>::WaveTableOsc(): phasor(0.), phaseIncr(0.),
    numWaveTables(0.){}

  template<class T>
  inline WaveTableOsc<T>::WaveTableOsc(int _maxWaveTables): WaveTableOsc(){
    maxWaveTables = _maxWaveTables;
  }

  template<class T>
  inline WaveTableOsc<T>::~WaveTableOsc(){
    for (int i=0; i < numWaveTables; i++){
      T *temp = waveTables[i].waveTable;
      if (temp != 0){
        delete[] temp;
      }
    }
  }

  template<class T>
  inline void WaveTableOsc<T>::setFreq(double incr){
    phaseIncr = incr;

    int waveTableIdx = 0;
    while((phaseIncr >= waveTables[waveTableIdx].topFreq) &&
          (waveTableIdx < (numWaveTables-1))){
      waveTableIdx++;
    }
    currWaveTable = waveTableIdx;
  }

  template<class T>
  inline T WaveTableOsc<T>::getOutput(){
    WaveTable<T>* table = &waveTables[currWaveTable];

    double temp = phasor * table->len;
    int intPart =  temp;
    double fracPart = temp - intPart;

    T samp0 = table->waveTable[intPart];
    T samp1 = table->waveTable[intPart+1];

    return samp0 + (samp1-samp0) * fracPart;
  }

  template<class T>
  inline T WaveTableOsc<T>::process(){
    updatePhase();
    return getOutput();
  }


  template<class T>
  inline int WaveTableOsc<T>::addWaveTable(int len, T* tableIn, double topFreq){
    if (numWaveTables < maxWaveTables){
      // float* table = wavetables[num_wavetables].wavetable = new float[len + 1];
      // wavetables[num_wavetables].len = len;
      // wavetables[num_wavetables].top_freq = top_freq;
      T* table = new T[len + 1];
      waveTables.push_back(WaveTable<T>({topFreq, len, table}));
      for (int i=0; i < len; i++){
        table[i] = tableIn[i];
      }
      table[len] = tableIn[0];
      numWaveTables++;

      return 0;
    }
    return numWaveTables;
  }

  template<class T>
  inline void setSineOsc(WaveTableOsc<T>& osc, int sampleRate, int overSample,
                         double baseFreq){
    int maxHarms = sampleRate / (3.0 * baseFreq) + 0.5;

    // rounding to next power of 2
    unsigned int v = maxHarms;
    v--; // in case already a power of 2
    v |= v >> 1; // fill in all bits to the right of the highest to 1
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16; // leaves 1's from highest bit -> location 0
    v++; // collapse all ones to leave a 1 at the highest bit

    int tableLen = v * 2 * overSample;
    double topFreq = baseFreq * 2. / sampleRate;
    double scale = 1.0;
    // double ar[table_len], ai[table_len]; // gets put into wavetable and deleted by ~WaveTableOsc
    T wave[tableLen];
    double freq = baseFreq;
    for (; maxHarms >=1; maxHarms >>=1){
      for (int i=0; i<tableLen; i++){
        wave[i] = 0.;
      }
      for (int i=0; i<tableLen; i++){
        wave[i] = scale * sin((T)i*2.*M_PI/tableLen);
      }

      if(osc.addWaveTable(tableLen, wave, topFreq)) scale = 0.;
      if (tableLen > constantRatioLimit)
        tableLen >>= 1;
      freq *= 2;
      topFreq *= 2;
    }

  }
}

#endif
