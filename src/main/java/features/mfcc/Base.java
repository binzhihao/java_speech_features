package features.mfcc;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static java.lang.Math.PI;

public class Base {

    public static INDArray mfcc(INDArray signal, double samplerate, double winlen, double winstep) {
        return mfcc(signal, samplerate, winlen, winstep, 13, 26, 0, 0, samplerate / 2,
                0.97, 22);
    }

    public static INDArray mfcc(INDArray signal, double samplerate, double winlen, double winstep, int numcep,
                                int nfilt, int nfft, double lowfreq, double highfreq, double preemph, int ceplifter) {
        if (signal == null || signal.isEmpty()) {
            throw new RuntimeException("No signal!");
        }
        if (nfft == 0) {
            nfft = calculateNfft(samplerate, winlen);
        }
        INDArray[] result = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph);
        INDArray feat = result[0];
        INDArray energy = result[0];
        feat = dct(feat);
        // slice

        lifter(feat, ceplifter);
        return feat;
    }

    private static int calculateNfft(double samplerate, double winlen) {
        double windowLengthSamples = winlen * samplerate;
        int nfft = 1;
        while (nfft < windowLengthSamples) {
            nfft *= 2;
        }
        return nfft;
    }

    private static INDArray[] fbank(INDArray signal, double samplerate, double winlen, double winstep,
                                    int nfilter, int nfft, double lowfreq, double highfreq, double preemph) {
        preemphasis(signal, preemph);
        INDArray frames = framesig(signal, winlen * samplerate, winstep * samplerate);
        INDArray pspec = powspec(frames, nfft);
        INDArray energy = pspec.sum();
        for (long i = 0; i < energy.length(); ++i) {
            if (energy.getDouble(i) == 0) {
                energy.putScalar(i, 0.0001);
            }
            // compute log here for convenience
            energy.putScalar(i, Math.log(energy.getDouble(i)));
        }
        INDArray fb = getFilterbanks(nfilter, nfft, samplerate, lowfreq, highfreq);
        INDArray feat = pspec.mmuli(fb.transposei());
        for (long i = 0; i < feat.length(); ++i) {
            if (feat.getDouble(i) == 0) {
                feat.putScalar(i, 0.0001);
            }
            // compute log here for convenience
            feat.putScalar(i, Math.log(feat.getDouble(i)));
        }
        return new INDArray[]{feat, energy};
    }

    private static void preemphasis(INDArray signal, double coeff) {
        for (long i = signal.length() - 1; i > 0; --i) {
            signal.putScalar(i, signal.getDouble(i) - coeff * signal.getDouble(i - 1));
        }
    }

    private static INDArray framesig(INDArray signal, double frameLength, double frameStep) {
        // Frame a signal into overlapping frames. But we apply the whole single as one frame, just return itself.
        return signal;
    }

    /**
     * Compute the power spectrum of each frame in frames.
     * If frames is an NxD matrix, output will be Nx(NFFT/2+1).
     *
     * @param frames the array of frames. Each row is a frame.
     * @param nfft   the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
     * @return Each row will be the power spectrum of the corresponding frame.
     */
    private static INDArray powspec(INDArray frames, int nfft) {
        INDArray result = Nd4j.zeros(frames.rows(), nfft / 2 + 1);
        /*IComplexNDArray complex_spec = FFT(frames, nfft);
        for (long i = 0; i < complex_spec.length(); ++i) {
            result.putScalar(i, 1.0 / (double) nfft *
                    (Math.abs(complex_spec.getReal(i)) + Math.pow(Math.abs(complex_spec.getImag(i))), 2));
        }*/
        return result;
    }

    private static INDArray getFilterbanks(int nfilt, int nfft, double samplerate, double lowfreq, double highfreq) {
        assert highfreq <= samplerate / 2;
        double lowmel = hz2mel(lowfreq);
        double highmel = hz2mel(highfreq);
        INDArray melpoints = Nd4j.linspace(lowmel, highmel, nfilt + 2, DataType.DOUBLE);
        INDArray bin = Nd4j.create(melpoints.length());
        for (long i = 0; i < melpoints.length(); ++i) {
            bin.putScalar(i, Math.floor((nfft + 1) * mel2hz(melpoints.getDouble(i)) / samplerate));
        }
        INDArray fbank = Nd4j.zeros(nfilt, nfft / 2 + 1);
        for (long j = 0; j < nfilt; ++j) {
            for (long i = (long) bin.getDouble(j); i < (long) bin.getDouble(j + 1); ++i) {
                fbank.putScalar(j, i, (i - bin.getDouble(j)) / (bin.getDouble(j + 1) - bin.getDouble(j)));
            }
            for (long i = (long) bin.getDouble(j + 1); i < (long) bin.getDouble(j + 2); ++i) {
                fbank.putScalar(j, i, (bin.getDouble(j + 2) - i) / (bin.getDouble(j + 2) - bin.getDouble(j + 1)));
            }
        }
        return fbank;
    }

    private static double hz2mel(double hz) {
        return 2595 * Math.log10(1 + hz / 700.);
    }

    private static double mel2hz(double mel) {
        return 700 * (Math.pow(10, mel / 2595.) - 1);
    }

    private static INDArray dct(INDArray feat) {
        int M = feat.rows();
        int N = feat.columns();
        INDArray feature = Nd4j.zeros(M, N);
        double f0 = Math.sqrt(1. / (4. * N));
        double f = Math.sqrt(1. / (2. * N));
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < N; ++k) {
                double sum = 0;
                for (int n = 0; n < N; ++n) {
                    sum += feat.getDouble(i, n) * Math.cos(PI * k * (2 * n + 1) / (2 * N));
                }
                sum *= 2;
                sum *= k == 0 ? f0 : f;
                feature.putScalar(i, k, sum);
            }
        }
        return feature;
    }

    private static void lifter(INDArray cepstra, int L) {
        if (L > 0) {
            /*nframes,ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L/2.)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra*/
        }
    }

}
