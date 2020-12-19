#include <random>
#include <limits>

namespace madigan{

  class XorShift128PlusBitShifterPseudoRandomBooleanGenerator {
  public:
    bool randBool() {
      if (counter == 0) {
        counter = sizeof(GeneratorType::result_type) * CHAR_BIT;
        random_integer = generator();
      }
      return (random_integer >> --counter) & 1;
    }

  private:
    class XorShift128Plus {
    public:
      using result_type = uint64_t;

      XorShift128Plus() {
        std::random_device rd;
        state[0] = rd();
        state[1] = rd();
      }

      result_type operator()() {
        auto x = state[0];
        auto y = state[1];
        state[0] = y;
        x ^= x << 23;
        state[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
        return state[1] + y;
      }

    private:
      result_type state[2];
    };

    using GeneratorType = XorShift128Plus;

    GeneratorType generator;
    GeneratorType::result_type random_integer;
    int counter = 0;
  };
}
